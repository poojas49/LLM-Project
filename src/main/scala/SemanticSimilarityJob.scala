import org.apache.hadoop.conf.Configuration
import org.apache.hadoop.fs.{FileSystem, Path}
import org.apache.hadoop.io.{LongWritable, Text}
import org.apache.hadoop.mapreduce.{Job, Mapper, Reducer}
import org.apache.hadoop.mapreduce.lib.input.FileInputFormat
import org.apache.hadoop.mapreduce.lib.output.FileOutputFormat
import org.apache.log4j.Logger
import com.typesafe.config.ConfigFactory

import scala.jdk.CollectionConverters._
import scala.util.{Try, Success, Failure}

/**
 * SemanticSimilarityJob object implements a Hadoop MapReduce job that calculates
 * semantic similarity between tokens based on their embeddings.
 *
 * Input: Token embeddings from the EmbeddingJob
 *        Format: <token>\t[<embedding_values>]
 *        Example: "hello\t[0.1, 0.2, 0.3, ...]"
 *
 * Output: Token with its top-K most similar tokens and their similarity scores
 *         Format: <token>\t<similar_token1>(score1),<similar_token2>(score2),...
 *         Example: "hello\thi(0.95),hey(0.92),greetings(0.88),..."
 *
 * Design Rationale:
 * 1. Configurability: Uses external configuration for easy parameter tuning.
 * 2. Scalability: Leverages Hadoop MapReduce for distributed processing of large datasets.
 * 3. Functional Approach: Utilizes immutable data structures and pure functions where possible.
 * 4. Error Handling: Implements comprehensive error handling using Try monad.
 * 5. Efficiency: Calculates similarities in a single pass over the data in the reducer.
 */
object SemanticSimilarityJob {
  private val logger = Logger.getLogger(getClass)
  private val config = ConfigFactory.load()

  // Configuration parameters
  private val jobName = config.getString("semantic-similarity.job-name")
  private val inputSplitDelimiter = config.getString("semantic-similarity.input-split-delimiter")
  private val embeddingDelimiter = config.getString("semantic-similarity.embedding-delimiter")
  private val topK = config.getInt("semantic-similarity.top-k")
  private val similarityFormat = config.getString("semantic-similarity.similarity-format")

  /**
   * Mapper class for the semantic similarity job.
   * Extracts token-embedding pairs from input and emits them with a constant key.
   *
   * Input: <LongWritable, Text>
   *   Key: Line offset (not used)
   *   Value: Input line in format <token>\t[<embedding_values>]
   * Output: <Text, Text>
   *   Key: Constant "all" (to ensure all data goes to a single reducer)
   *   Value: Token and its embedding, separated by embeddingDelimiter
   *
   * Design Rationale:
   * 1. Simple Extraction: Focuses on parsing input and emitting relevant data.
   * 2. Single Reducer: Uses a constant key to ensure all data is processed together.
   * 3. Error Logging: Logs malformed input for debugging purposes.
   */
  class SimilarityMapper extends Mapper[LongWritable, Text, Text, Text] {
    override def map(key: LongWritable, value: Text, context: Mapper[LongWritable, Text, Text, Text]#Context): Unit = {
      value.toString.split(inputSplitDelimiter) match {
        case Array(token, embedding) =>
          context.write(new Text("all"), new Text(s"$token$embeddingDelimiter$embedding"))
        case _ =>
          logger.warn(s"Malformed input: ${value.toString}")
      }
    }
  }

  /**
   * Reducer class for the semantic similarity job.
   * Calculates cosine similarity between token embeddings and finds the top-K similar tokens for each token.
   *
   * Input: <Text, Text>
   *   Key: Constant "all"
   *   Value: Iterable of token-embedding pairs
   * Output: <Text, Text>
   *   Key: Token
   *   Value: Comma-separated list of top-K similar tokens with their similarity scores
   *
   * Design Rationale:
   * 1. In-memory Processing: Loads all embeddings into memory for efficient pairwise comparison.
   * 2. Functional Approach: Uses immutable data structures and pure functions for clarity and thread-safety.
   * 3. Modular Design: Splits the process into smaller, focused functions for better maintainability.
   */
  class SimilarityReducer extends Reducer[Text, Text, Text, Text] {
    override def reduce(key: Text, values: java.lang.Iterable[Text], context: Reducer[Text, Text, Text, Text]#Context): Unit = {
      val embeddings = parseEmbeddings(values)
      val similarities = calculateSimilarities(embeddings)
      writeSimilarities(similarities, context)
    }

    /**
     * Parses the input values into a map of tokens to their embeddings.
     *
     * @param values Iterable of token-embedding pairs
     * @return Map of tokens to their embeddings
     */
    private def parseEmbeddings(values: java.lang.Iterable[Text]): Map[String, Array[Float]] = {
      values.asScala.foldLeft(Map.empty[String, Array[Float]]) { (acc, value) =>
        value.toString.split(s"\\$embeddingDelimiter") match {
          case Array(token, embeddingStr) =>
            val embedding = embeddingStr.stripPrefix("[").stripSuffix("]").split(",").map(_.trim.toFloat)
            acc + (token -> embedding)
          case _ => acc
        }
      }
    }

    /**
     * Calculates pairwise similarities between all tokens.
     *
     * @param embeddings Map of tokens to their embeddings
     * @return Map of tokens to their top-K similar tokens with similarity scores
     */
    private def calculateSimilarities(embeddings: Map[String, Array[Float]]): Map[String, Seq[(String, Double)]] = {
      embeddings.map { case (token1, embedding1) =>
        val tokenSimilarities = embeddings.filter(_._1 != token1).map { case (token2, embedding2) =>
          (token2, cosineSimilarity(embedding1, embedding2))
        }.toSeq.sortBy(-_._2).take(topK)
        (token1, tokenSimilarities)
      }
    }

    /**
     * Writes the calculated similarities to the context.
     *
     * @param similarities Map of tokens to their similar tokens with scores
     * @param context Reducer context for writing output
     */
    private def writeSimilarities(similarities: Map[String, Seq[(String, Double)]], context: Reducer[Text, Text, Text, Text]#Context): Unit = {
      similarities.foreach { case (token, similarTokens) =>
        val formattedSimilarities = similarTokens.map { case (similarToken, similarity) =>
          s"$similarToken(${similarity.formatted(similarityFormat)})"
        }.mkString(",")
        context.write(new Text(token), new Text(formattedSimilarities))
      }
    }

    /**
     * Calculates the cosine similarity between two vectors.
     *
     * @param vec1 First vector
     * @param vec2 Second vector
     * @return Cosine similarity as a Double
     *
     * Design Rationale:
     * 1. Efficiency: Calculates dot product and magnitudes in a single pass.
     * 2. Numerical Stability: Uses separate accumulation for dot product and magnitudes.
     */
    private def cosineSimilarity(vec1: Array[Float], vec2: Array[Float]): Double = {
      require(vec1.length == vec2.length, "Vectors must have the same length")
      val (dotProduct, mag1, mag2) = (vec1 zip vec2).foldLeft((0.0, 0.0, 0.0)) {
        case ((dot, mag1Sq, mag2Sq), (v1, v2)) =>
          (dot + v1 * v2, mag1Sq + v1 * v1, mag2Sq + v2 * v2)
      }
      dotProduct / (math.sqrt(mag1) * math.sqrt(mag2))
    }
  }

  /**
   * Runs the semantic similarity job.
   *
   * @param conf Hadoop Configuration
   * @param input Input path
   * @param output Output path
   * @return Try[Unit] representing success or failure of the job
   *
   * Design Rationale:
   * 1. Comprehensive Error Handling: Uses Try for composable error handling.
   * 2. Separation of Concerns: Divides job setup, output cleanup, and job execution into separate steps.
   * 3. Logging: Provides detailed logging for job start, completion, and failure.
   */
  def runJob(conf: Configuration, input: Path, output: Path): Try[Unit] = {
    val setupJob = Try(Job.getInstance(conf, jobName)).map { job =>
      job.setJarByClass(this.getClass)
      job.setMapperClass(classOf[SimilarityMapper])
      job.setReducerClass(classOf[SimilarityReducer])
      job.setOutputKeyClass(classOf[Text])
      job.setOutputValueClass(classOf[Text])
      FileInputFormat.addInputPath(job, input)
      FileOutputFormat.setOutputPath(job, output)
      job
    }

    val cleanupOutput = setupJob.flatMap(_ => Try(FileSystem.get(conf))).map { fs =>
      if (fs.exists(output)) fs.delete(output, true)
    }

    val runAndComplete = cleanupOutput.flatMap { _ =>
      Try {
        logger.info(s"Starting $jobName job with input: $input and output: $output")
        setupJob.get.waitForCompletion(true)
      }.flatMap { completed =>
        if (completed) Success(())
        else Failure(new Exception(s"$jobName job failed"))
      }
    }

    runAndComplete.map { _ =>
      logger.info(s"$jobName job completed successfully")
    }
  }
}