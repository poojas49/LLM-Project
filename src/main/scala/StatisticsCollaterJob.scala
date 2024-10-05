import org.apache.hadoop.conf.Configuration
import org.apache.hadoop.fs.{FileSystem, Path}
import org.apache.hadoop.io.{LongWritable, Text}
import org.apache.hadoop.mapreduce.lib.input.{MultipleInputs, TextInputFormat}
import org.apache.hadoop.mapreduce.lib.output.FileOutputFormat
import org.apache.hadoop.mapreduce.{Job, Mapper, Reducer}
import org.yaml.snakeyaml.Yaml
import com.typesafe.config.ConfigFactory

import java.io.StringWriter
import scala.jdk.CollectionConverters.*

/**
 * StatisticsCollaterJob object implements a Hadoop MapReduce job that collates statistics
 * from multiple sources (tokenization and similarity data) into a single, structured output.
 * This job serves as the final step in a text processing pipeline, combining and formatting
 * the results for easy consumption.
 *
 * Inputs:
 * 1. Tokenization data: <word>\t<token>\t<frequency>
 * 2. Similarity data: <token>\t<similar_token1>(score1),<similar_token2>(score2),...
 *
 * Output: YAML-formatted statistics for each token, including:
 * - Original word
 * - Integer token representation
 * - Frequency
 * - List of semantically similar tokens with similarity scores
 *
 * Design Rationale:
 * 1. Configurability: Use of external configuration allows for easy parameter tuning.
 * 2. Multiple Input Sources: Utilizes Hadoop's MultipleInputs to process data from different stages of the pipeline.
 * 3. Structured Output: Uses YAML for a human-readable and machine-parseable output format.
 * 4. Flexibility: The design allows for easy addition of new statistics or data sources.
 * 5. Scalability: Leverages Hadoop's distributed processing capabilities for large-scale data collation.
 */
object StatisticsCollaterJob {
  // Load configuration values
  private val config = ConfigFactory.load()
  private val jobName = config.getString("statistics-collater.job-name")
  private val inputSplitDelimiter = config.getString("statistics-collater.input-split-delimiter")
  private val similarityPairDelimiter = config.getString("statistics-collater.similarity-pair-delimiter")
  private val similarityScoreDelimiter = config.getString("statistics-collater.similarity-score-delimiter")
  private val naValue = config.getString("statistics-collater.na-value")

  // Configuration for YAML output keys
  private val wordKey = config.getString("yaml-output.word-key")
  private val intTokenKey = config.getString("yaml-output.int-token-key")
  private val frequencyKey = config.getString("yaml-output.frequency-key")
  private val similarTokensKey = config.getString("yaml-output.similar-tokens-key")

  /**
   * Mapper class for processing tokenization data.
   * Extracts word, token, and frequency information from the tokenization job output.
   *
   * Input: <LongWritable, Text>
   *   - Key: Line offset in input file (not used)
   *   - Value: Line of text in format <word>\t<token>\t<frequency>
   * Output: <Text, Text>
   *   - Key: Token
   *   - Value: Word and frequency, separated by inputSplitDelimiter
   *
   * Design Rationale:
   * 1. Simple Extraction: Focuses solely on parsing and emitting tokenization data.
   * 2. Key-Value Pairing: Uses the token as the key to allow merging with similarity data in the reducer.
   * 3. Minimal Processing: Keeps mapper light, pushing main aggregation to the reducer.
   */
  class TokenizationMapper extends Mapper[LongWritable, Text, Text, Text] {
    override def map(key: LongWritable, value: Text, context: Mapper[LongWritable, Text, Text, Text]#Context): Unit = {
      val parts = value.toString.split(inputSplitDelimiter)
      if (parts.length == 3) {
        val word = parts(0)
        val token = parts(1)
        val frequency = parts(2)
        context.write(new Text(token), new Text(s"$word$inputSplitDelimiter$frequency"))
      }
    }
  }

  /**
   * Mapper class for processing similarity data.
   * Extracts token and its similar tokens from the similarity job output.
   *
   * Input: <LongWritable, Text>
   *   - Key: Line offset in input file (not used)
   *   - Value: Line of text in format <token>\t<similar_token1>(score1),<similar_token2>(score2),...
   * Output: <Text, Text>
   *   - Key: Token
   *   - Value: Similar tokens with scores, as received in input
   *
   * Design Rationale:
   * 1. Simple Extraction: Focuses solely on parsing and emitting similarity data.
   * 2. Key-Value Pairing: Uses the token as the key to allow merging with tokenization data in the reducer.
   * 3. Preservation of Data: Keeps similarity data intact for processing in the reducer.
   */
  class SimilarityMapper extends Mapper[LongWritable, Text, Text, Text] {
    override def map(key: LongWritable, value: Text, context: Mapper[LongWritable, Text, Text, Text]#Context): Unit = {
      val parts = value.toString.split(inputSplitDelimiter)
      if (parts.length == 2) {
        val token = parts(0)
        val similarities = parts(1)
        val similarityPairs = similarities.split(similarityPairDelimiter).map(_.trim)
        context.write(new Text(token), new Text(similarityPairs.mkString(similarityPairDelimiter)))
      }
    }
  }

  /**
   * Reducer class that combines tokenization and similarity data, and outputs a YAML-formatted result.
   *
   * Input: <Text, Text>
   *   - Key: Token
   *   - Value: Either word and frequency (from TokenizationMapper) or similarity data (from SimilarityMapper)
   * Output: <Text, Text>
   *   - Key: null (not used)
   *   - Value: YAML-formatted string containing all statistics for the token
   *
   * Design Rationale:
   * 1. Data Aggregation: Combines data from multiple sources (tokenization and similarity) for each token.
   * 2. Flexible Processing: Uses a fold operation to handle varying input types and accumulate results.
   * 3. Structured Output: Generates a YAML-formatted output for easy parsing and readability.
   * 4. Null Key Output: Allows for sequential, unkeyed output of YAML data.
   */
  class StatisticsReducer extends Reducer[Text, Text, Text, Text] {
    override def reduce(key: Text, values: java.lang.Iterable[Text], context: Reducer[Text, Text, Text, Text]#Context): Unit = {
      val token = key.toString

      // Aggregate data from both mappers
      val (word, frequency, similarities) = values.asScala.foldLeft(("", 0L, List.empty[String])) {
        case ((word, frequency, similarities), value) =>
          val parts = value.toString.split(inputSplitDelimiter)
          if (parts.length == 2) {
            // Processing tokenization data
            (parts(0), parts(1).toLong, similarities)
          } else {
            // Processing similarity data
            val similarityPairs = value.toString.split(similarityPairDelimiter).map { pair =>
              val Array(similarToken, score) = pair.split(s"\\$similarityScoreDelimiter")
              s"$similarToken"
            }.toList
            (word, frequency, similarityPairs)
          }
      }

      // Prepare data for YAML output
      val data = Map(
        wordKey -> word,
        intTokenKey -> token,
        frequencyKey -> frequency,
        similarTokensKey -> (if (similarities.isEmpty) naValue else similarities.asJava)
      )

      // Generate YAML output
      val yaml = new Yaml()
      val writer = new StringWriter()
      yaml.dump(data.asJava, writer)

      context.write(null, new Text(writer.toString))
    }
  }

  /**
   * Runs the statistics collation job.
   * This method sets up and executes the Hadoop MapReduce job with multiple input sources.
   *
   * @param conf The Hadoop configuration
   * @param tokenizationInput Path to the tokenization job output
   * @param similarityInput Path to the similarity job output
   * @param output Path for the final collated statistics output
   *
   * Design Rationale:
   * 1. Multiple Inputs: Uses MultipleInputs to process data from different stages of the pipeline.
   * 2. Output Cleanup: Ensures clean job runs by removing any existing output.
   * 3. Error Handling: Exits with non-zero status on job failure for error detection in wrapper scripts.
   * 4. Single Reducer: Uses a single reducer to ensure all data for each token is processed together.
   */
  def runJob(conf: Configuration, tokenizationInput: Path, similarityInput: Path, output: Path): Unit = {
    val job = Job.getInstance(conf, jobName)
    job.setJarByClass(this.getClass)
    job.setReducerClass(classOf[StatisticsReducer])
    job.setOutputKeyClass(classOf[Text])
    job.setOutputValueClass(classOf[Text])

    // Set up multiple input paths with different mappers
    MultipleInputs.addInputPath(job, tokenizationInput, classOf[TextInputFormat], classOf[TokenizationMapper])
    MultipleInputs.addInputPath(job, similarityInput, classOf[TextInputFormat], classOf[SimilarityMapper])

    FileOutputFormat.setOutputPath(job, output)

    // Delete output path if it exists
    val fs = FileSystem.get(conf)
    if (fs.exists(output)) {
      fs.delete(output, true)
    }

    // Run the job and exit with error if it fails
    if (!job.waitForCompletion(true)) System.exit(1)
  }
}