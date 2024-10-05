import org.apache.hadoop.conf.Configuration
import org.apache.hadoop.fs.{FileSystem, Path}
import org.apache.hadoop.io.{LongWritable, Text}
import org.apache.hadoop.mapreduce.{Job, Mapper, Reducer}
import org.apache.hadoop.mapreduce.lib.input.FileInputFormat
import org.apache.hadoop.mapreduce.lib.output.FileOutputFormat
import org.apache.log4j.Logger
import com.typesafe.config.ConfigFactory

import scala.jdk.CollectionConverters.*

/**
 * SlidingWindowJob object implements a Hadoop MapReduce job that processes input data
 * using a sliding window approach. This is typically used in natural language processing
 * tasks where context is important, such as language modeling or feature extraction.
 *
 * Input: Tab-separated values (TSV) file with format: <word>\t<token>\t<frequency>
 * Output: Two types of key-value pairs:
 *         1. Key: "input_<index>", Value: Comma-separated list of tokens (the input window)
 *         2. Key: "label_<index>", Value: Single token (the label for the corresponding input window)
 *
 * Design Rationale:
 * 1. Distributed Processing: Uses MapReduce to handle large-scale token data efficiently.
 * 2. Configurability: Uses external configuration for easy parameter adjustment (e.g., window size).
 * 3. Context Preservation: Sliding window approach maintains contextual information for each token.
 * 4. Flexible Output: Separate input and label outputs allow for various downstream tasks (e.g., language modeling, classification).
 */
object SlidingWindowJob {
  // Initialize logger for this class
  private val logger = Logger.getLogger(getClass)

  // Load configuration from application.conf file
  private val config = ConfigFactory.load()

  // Retrieve specific configuration values
  private val windowSize = config.getInt("sliding-window.window-size")
  private val jobName = config.getString("sliding-window.job-name")
  private val inputSplitDelimiter = config.getString("sliding-window.input-split-delimiter")
  private val tokenIndex = config.getInt("sliding-window.token-index")

  /**
   * Mapper class for the sliding window job.
   * It extracts tokens from input lines and emits them for further processing.
   *
   * Input: <LongWritable, Text>
   *   - Key: Line offset in the input file (not used)
   *   - Value: Line of text from the input file (TSV format: <word>\t<token>\t<frequency>)
   * Output: <Text, Text>
   *   - Key: "data" (constant key to ensure all tokens go to the same reducer)
   *   - Value: The extracted token
   *
   * Design Rationale:
   * - Single key output ensures all tokens are processed together in the reducer.
   * - Extracts only the token, discarding word and frequency, as context is built in the reducer.
   */
  private class SlidingWindowMapper extends Mapper[LongWritable, Text, Text, Text] {
    override def map(key: LongWritable, value: Text, context: Mapper[LongWritable, Text, Text, Text]#Context): Unit = {
      val inputLine = value.toString
      logger.info(s"Processing input line: $inputLine")

      val parts = inputLine.split(inputSplitDelimiter)
      if (parts.length > tokenIndex) {
        val token = parts(tokenIndex)
        logger.info(s"Extracted token: $token")
        // Emit all tokens with the same key "data" to ensure they go to the same reducer
        context.write(new Text("data"), new Text(token))
      } else {
        logger.warn(s"Skipping input line due to insufficient parts: $inputLine")
      }
    }
  }

  /**
   * Reducer class for the sliding window job.
   * It processes the tokens using a sliding window approach and emits window-label pairs.
   *
   * Input: <Text, Text>
   *   - Key: "data" (constant key from mapper)
   *   - Value: Iterable of all tokens
   * Output: <Text, Text>
   *   - Two types of output:
   *     1. Key: "input_<index>", Value: Comma-separated list of tokens (the input window)
   *     2. Key: "label_<index>", Value: Single token (the label for the corresponding input window)
   *
   * Design Rationale:
   * - Sliding window approach captures context for each token.
   * - Separate input and label outputs allow for flexible use in various ML tasks.
   * - Index in key maintains order and allows for reconstruction of original sequence if needed.
   */
  private class SlidingWindowReducer extends Reducer[Text, Text, Text, Text] {
    override def reduce(key: Text, values: java.lang.Iterable[Text], context: Reducer[Text, Text, Text, Text]#Context): Unit = {
      val tokens = values.asScala.map(_.toString).toArray
      logger.info(s"Reducer received ${tokens.length} tokens")

      // Process tokens using a sliding window approach
      tokens.sliding(windowSize + 1).zipWithIndex.foreach { case (window, idx) =>
        if (window.length == windowSize + 1) {
          val (inputWindow, label) = window.splitAt(windowSize)
          val inputStr = inputWindow.mkString(",")
          val labelStr = label.head
          logger.info(s"Emitting window $idx: input=$inputStr, label=$labelStr")
          // Emit both the input window and its corresponding label
          context.write(new Text(s"input_$idx"), new Text(inputStr))
          context.write(new Text(s"label_$idx"), new Text(labelStr))
        } else {
          logger.warn(s"Skipping window $idx due to insufficient length: ${window.mkString(",")}")
        }
      }
    }
  }

  /**
   * Runs the sliding window job.
   * This method sets up and executes the Hadoop MapReduce job.
   *
   * @param conf The Hadoop configuration
   * @param input The input path for the job (TSV file with word, token, and frequency)
   * @param output The output path for the job (Text files with input windows and labels)
   *
   * Design Rationale:
   * - Standard Hadoop job setup for consistency with other pipeline stages.
   * - Output cleanup ensures clean runs but should be used cautiously in production.
   * - Detailed logging for monitoring job progress and debugging.
   * - Non-zero exit on failure allows for error detection in the overall pipeline.
   */
  def runJob(conf: Configuration, input: Path, output: Path): Unit = {
    val job = Job.getInstance(conf, jobName)
    job.setJarByClass(this.getClass)
    job.setMapperClass(classOf[SlidingWindowMapper])
    job.setReducerClass(classOf[SlidingWindowReducer])
    job.setMapOutputKeyClass(classOf[Text])
    job.setMapOutputValueClass(classOf[Text])
    job.setOutputKeyClass(classOf[Text])
    job.setOutputValueClass(classOf[Text])
    FileInputFormat.addInputPath(job, input)
    FileOutputFormat.setOutputPath(job, output)

    // Delete output path if it exists
    val fs = FileSystem.get(conf)
    if (fs.exists(output)) {
      fs.delete(output, true)
    }

    logger.info(s"Starting Sliding Window job with input: $input and output: $output")
    if (!job.waitForCompletion(true)) {
      logger.error("Sliding Window job failed")
      System.exit(1)
    }
    logger.info("Sliding Window job completed successfully")
  }
}