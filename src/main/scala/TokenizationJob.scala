import org.apache.hadoop.conf.Configuration
import org.apache.hadoop.fs.{FileSystem, Path}
import org.apache.hadoop.io.{IntWritable, LongWritable, Text}
import org.apache.hadoop.mapreduce.{Job, Mapper, Reducer}
import org.apache.hadoop.mapreduce.lib.input.FileInputFormat
import org.apache.hadoop.mapreduce.lib.output.FileOutputFormat
import com.knuddels.jtokkit.Encodings
import com.knuddels.jtokkit.api.{IntArrayList, ModelType}
import com.typesafe.config.ConfigFactory
import org.apache.log4j.Logger

import scala.jdk.CollectionConverters.*

/**
 * TokenizationJob object handles the tokenization of input text using a specified encoding model.
 * It's designed as a Hadoop MapReduce job, allowing for distributed processing of large text datasets.
 *
 * Input: Raw text files
 * Output: Tab-separated values (TSV) file with format: <word>\t<token>\t<frequency>
 *
 * Design Rationale:
 * 1. Distributed Processing: Uses MapReduce to handle large-scale text data efficiently.
 * 2. Configurability: Uses external configuration for easy parameter adjustment.
 * 3. Preprocessing: Applies text normalization before tokenization for consistency.
 * 4. Tokenization: Uses JTokkit library for flexible and model-specific tokenization.
 * 5. Frequency Counting: Aggregates token frequencies, useful for further analysis.
 */
object TokenizationJob {
  private val logger = Logger.getLogger(this.getClass)

  // Load configuration from application.conf file
  private val config = ConfigFactory.load()
  // Retrieve specific configuration values
  private val modelTypeString = config.getString("tokenization.model-type")
  private val jobName = config.getString("tokenization.job-name")
  private val preprocessingRegex = config.getString("tokenization.preprocessing-regex")

  logger.info(s"TokenizationJob initialized with model type: $modelTypeString, job name: $jobName")
  logger.debug(s"Preprocessing regex: $preprocessingRegex")

  /**
   * Mapper class for the tokenization job.
   * It preprocesses the input text, tokenizes words, and emits word-token pairs with a count of 1.
   *
   * Input: <LongWritable, Text>
   *   - Key: Line offset in the input file (not used)
   *   - Value: Line of text from the input file
   * Output: <Text, IntWritable>
   *   - Key: Tab-separated word and token (e.g., "hello\t1234")
   *   - Value: Count (always 1 in this case)
   */
  private class TokenizationMapper extends Mapper[LongWritable, Text, Text, IntWritable] {
    private val logger = Logger.getLogger(this.getClass)

    // Lazy initialization of the encoding to ensure it's only created when needed
    private lazy val encoding = {
      logger.info("Initializing encoding model")
      val registry = Encodings.newDefaultEncodingRegistry()
      registry.getEncodingForModel(ModelType.valueOf(modelTypeString))
    }
    // Reusable IntWritable object to avoid unnecessary object creation
    private val one = new IntWritable(1)

    /**
     * Preprocesses the input text by converting to lowercase, removing unwanted characters,
     * and normalizing whitespace.
     *
     * Design Rationale:
     * - Normalization ensures consistent tokenization across different text inputs.
     * - Configurable regex allows for flexible preprocessing rules.
     *
     * @param text The input text to preprocess
     * @return The preprocessed text
     */
    private def preprocess(text: String): String = {
      logger.debug(s"Preprocessing text: ${text.take(100)}...")
      val result = text.toLowerCase
        .replaceAll(preprocessingRegex, "") // Remove all characters except lowercase letters, numbers, and spaces
        .split("\\s+") // Split by whitespace
        .mkString(" ") // Join back with single spaces
      logger.debug(s"Preprocessed text: ${result.take(100)}...")
      result
    }

    /**
     * The map function processes each line of input text.
     * It tokenizes each word and emits word-token pairs with a count of 1.
     *
     * Design Rationale:
     * - Word-level tokenization allows for analysis of both words and subword tokens.
     * - Emitting counts of 1 allows the reducer to aggregate frequencies.
     */
    override def map(key: LongWritable, value: Text, context: Mapper[LongWritable, Text, Text, IntWritable]#Context): Unit = {
      logger.debug(s"Processing input line: ${value.toString.take(100)}...")
      val text = preprocess(value.toString)
      val words = text.split("\\s+") // Split by whitespace

      logger.debug(s"Tokenizing ${words.length} words")
      words.foreach { word =>
        val encodedTokens = encoding.encode(word)
        encodedTokens.toArray.foreach { token =>
          val tokenList = new IntArrayList(1)
          tokenList.add(token)
          val decodedWord = encoding.decode(tokenList)
          context.write(new Text(s"$decodedWord\t$token"), one)
          logger.trace(s"Emitted: $decodedWord\t$token\t1")
        }
      }
    }
  }

  /**
   * Reducer class for the tokenization job.
   * It sums up the counts for each word-token pair and formats the output.
   *
   * Input: <Text, IntWritable>
   *   - Key: Tab-separated word and token (e.g., "hello\t1234")
   *   - Value: Iterable of counts (1s from the mapper)
   * Output: <Text, Text>
   *   - Key: null (not used)
   *   - Value: Tab-separated word, token, and frequency (e.g., "hello\t1234\t5")
   *
   * Design Rationale:
   * - Aggregation of frequencies provides valuable information for further analysis.
   * - Output format is designed for easy parsing in subsequent pipeline stages.
   */
  private class TokenizationReducer extends Reducer[Text, IntWritable, Text, Text] {
    private val logger = Logger.getLogger(this.getClass)

    override def reduce(key: Text, values: java.lang.Iterable[IntWritable], context: Reducer[Text, IntWritable, Text, Text]#Context): Unit = {
      val sum = values.asScala.map(_.get).sum
      val Array(word, token) = key.toString.split("\t")
      val output = s"$word\t$token\t$sum"
      context.write(null, new Text(output))
      logger.debug(s"Reduced: $output")
    }
  }

  /**
   * Runs the tokenization job.
   * This method sets up and executes the Hadoop MapReduce job.
   *
   * @param conf The Hadoop configuration
   * @param input The input path for the job (raw text files)
   * @param output The output path for the job (TSV file with word, token, and frequency)
   *
   * Design Rationale:
   * - Standard Hadoop job setup for consistency with other pipeline stages.
   * - Output cleanup ensures clean runs but should be used cautiously in production.
   * - Non-zero exit on failure allows for error detection in the overall pipeline.
   */
  def runJob(conf: Configuration, input: Path, output: Path): Unit = {
    logger.info(s"Starting tokenization job. Input: $input, Output: $output")

    val job = Job.getInstance(conf, jobName)
    job.setJarByClass(this.getClass)
    job.setMapperClass(classOf[TokenizationMapper])
    job.setReducerClass(classOf[TokenizationReducer])
    job.setOutputKeyClass(classOf[Text])
    job.setOutputValueClass(classOf[IntWritable])
    FileInputFormat.addInputPath(job, input)
    FileOutputFormat.setOutputPath(job, output)

    // Delete output path if it exists
    val fs = FileSystem.get(conf)
    if (fs.exists(output)) {
      logger.warn(s"Output path $output already exists. Deleting.")
      fs.delete(output, true)
    }

    logger.info("Submitting tokenization job")
    val success = job.waitForCompletion(true)
    if (success) {
      logger.info("Tokenization job completed successfully")
    } else {
      logger.error("Tokenization job failed")
      System.exit(1)
    }
  }
}