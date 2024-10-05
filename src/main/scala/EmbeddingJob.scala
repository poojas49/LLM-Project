import org.apache.hadoop.conf.Configuration
import org.apache.hadoop.fs.{FileSystem, Path}
import org.apache.hadoop.mapreduce.{Job, Mapper, Reducer}
import org.apache.hadoop.mapreduce.lib.input.FileInputFormat
import org.apache.hadoop.mapreduce.lib.output.FileOutputFormat
import org.apache.log4j.Logger
import org.deeplearning4j.nn.conf.NeuralNetConfiguration
import org.deeplearning4j.nn.conf.layers.{DenseLayer, OutputLayer}
import org.nd4j.linalg.activations.Activation
import org.nd4j.linalg.learning.config.Adam
import org.nd4j.linalg.lossfunctions.LossFunctions
import org.nd4j.linalg.api.ndarray.INDArray
import com.typesafe.config.ConfigFactory
import org.apache.hadoop.io.{LongWritable, Text}
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork
import org.nd4j.linalg.factory.Nd4j

import scala.jdk.CollectionConverters.*
import scala.util.{Failure, Success, Try}

/**
 * EmbeddingJob object implements a Hadoop MapReduce job that generates embeddings for tokens
 * using a neural network approach. This is typically used in natural language processing
 * tasks to create vector representations of words or tokens.
 *
 * Input: Token windows from the SlidingWindowJob, format: <key_type>\t<comma_separated_tokens>
 * Output: Token embeddings, format: <token>\t[<embedding_vector>]
 *
 * Design Rationale:
 * 1. Configurability: Uses external configuration for easy parameter tuning.
 * 2. Scalability: Leverages Hadoop MapReduce for distributed processing of large datasets.
 * 3. Neural Network Approach: Employs a simple autoencoder for generating embeddings.
 * 4. Functional Programming: Utilizes immutable data structures and pure functions where possible.
 * 5. Error Handling: Implements comprehensive error handling using Try monad.
 */
object EmbeddingJob {
  private val logger = Logger.getLogger(getClass)
  private val config = ConfigFactory.load()

  // Configuration parameters
  private val windowSize = config.getInt("embedding-job.window-size")
  private val embeddingSize = config.getInt("embedding-job.embedding-size")
  private val jobName = config.getString("embedding-job.job-name")
  private val inputSplitDelimiter = config.getString("embedding-job.input-split-delimiter")

  logger.info(s"EmbeddingJob initialized with window size: $windowSize, embedding size: $embeddingSize, job name: $jobName")
  logger.debug(s"Input split delimiter: $inputSplitDelimiter")

  /**
   * Mapper class for the embedding job.
   * Extracts token windows from input and emits them for processing by the reducer.
   *
   * Input: <LongWritable, Text>
   *   Key: Line offset (not used)
   *   Value: Input line in format <key_type>\t<comma_separated_tokens>
   * Output: <Text, Text>
   *   Key: Key type (e.g., "input" or "label")
   *   Value: Comma-separated tokens
   *
   * Design Rationale:
   * 1. Simple Extraction: Focuses on parsing input and emitting relevant data.
   * 2. Error Logging: Logs malformed input for debugging purposes.
   */
  class EmbeddingMapper extends Mapper[LongWritable, Text, Text, Text] {
    private val mapperLogger = Logger.getLogger(this.getClass)

    override def setup(context: Mapper[LongWritable, Text, Text, Text]#Context): Unit = {
      mapperLogger.info("EmbeddingMapper setup started")
    }

    override def map(key: LongWritable, value: Text, context: Mapper[LongWritable, Text, Text, Text]#Context): Unit = {
      val parts = value.toString.split(inputSplitDelimiter)
      parts match {
        case Array(keyType, tokens) =>
          val newKey = keyType.split("_")(0)
          context.write(new Text(newKey), new Text(tokens))
          mapperLogger.debug(s"Mapped key: $newKey, tokens: ${tokens.take(50)}...")
        case _ =>
          mapperLogger.warn(s"Malformed input: ${value.toString}")
      }
    }

    override def cleanup(context: Mapper[LongWritable, Text, Text, Text]#Context): Unit = {
      mapperLogger.info("EmbeddingMapper cleanup completed")
    }
  }

  /**
   * Reducer class for the embedding job.
   * Processes token windows using a neural network to generate embeddings.
   *
   * Input: <Text, Text>
   *   Key: Key type (e.g., "input" or "label")
   *   Value: Iterable of comma-separated token lists
   * Output: <Text, Text>
   *   Key: Token
   *   Value: Embedding vector as a string
   *
   * Design Rationale:
   * 1. Lazy Model Initialization: Creates the neural network only when needed.
   * 2. Functional Approach: Uses immutable data structures and pure functions for token processing.
   * 3. Averaging: Computes average embeddings for tokens that appear in multiple windows.
   */
  private class EmbeddingReducer extends Reducer[Text, Text, Text, Text] {
    private val reducerLogger = Logger.getLogger(this.getClass)
    private lazy val model: MultiLayerNetwork = createModel()

    override def setup(context: Reducer[Text, Text, Text, Text]#Context): Unit = {
      reducerLogger.info("EmbeddingReducer setup started")
      // Initialize the model in setup
      model
      reducerLogger.info("Neural network model initialized")
    }

    override def reduce(key: Text, values: java.lang.Iterable[Text], context: Reducer[Text, Text, Text, Text]#Context): Unit = {
      reducerLogger.info(s"Processing key: ${key.toString}")
      val allTokens = values.asScala.flatMap(_.toString.split(",").map(_.toInt)).toList
      reducerLogger.debug(s"Received ${allTokens.size} tokens for processing")

      val tokenEmbeddings = processTokens(allTokens)
      writeEmbeddings(tokenEmbeddings, context)
    }

    /**
     * Processes tokens to generate embeddings.
     *
     * @param tokens List of integer tokens
     * @return Map of tokens to their list of embeddings
     *
     * Design Rationale:
     * 1. Sliding Window: Processes tokens in windows to maintain context.
     * 2. Accumulation: Builds up embeddings for each token across all windows.
     * 3. Immutability: Uses immutable Map and functional fold for thread-safety.
     */
    private def processTokens(tokens: List[Int]): Map[Int, List[Array[Float]]] = {
      reducerLogger.info(s"Processing ${tokens.size} tokens")
      tokens.sliding(windowSize).zipWithIndex.foldLeft(Map.empty[Int, List[Array[Float]]]) { case (acc, (window, idx)) =>
        if (window.length == windowSize) {
          val inputFeature = createInputFeature(window)
          model.fit(inputFeature, inputFeature)
          reducerLogger.debug(s"Fitted model for window $idx: ${window.mkString(",")}")

          window.zipWithIndex.foldLeft(acc) { case (innerAcc, (token, index)) =>
            val embedding = model.getLayer(0).getParam("W").getColumn(index).toFloatVector
            reducerLogger.trace(s"Generated embedding for token $token")
            innerAcc.updated(token, embedding :: innerAcc.getOrElse(token, List.empty))
          }
        } else {
          reducerLogger.warn(s"Skipping window $idx due to insufficient length: ${window.mkString(",")}")
          acc
        }
      }
    }

    /**
     * Writes the final embeddings to the context.
     *
     * @param tokenEmbeddings Map of tokens to their list of embeddings
     * @param context Reducer context for writing output
     *
     * Design Rationale:
     * 1. Averaging: Computes the mean embedding for each token.
     * 2. Formatting: Converts embeddings to a string format for output.
     */
    private def writeEmbeddings(tokenEmbeddings: Map[Int, List[Array[Float]]], context: Reducer[Text, Text, Text, Text]#Context): Unit = {
      reducerLogger.info(s"Writing embeddings for ${tokenEmbeddings.size} tokens")
      tokenEmbeddings.foreach { case (token, embeddings) =>
        val avgEmbedding = embeddings.transpose.map(v => v.sum / v.length)
        val embeddingStr = avgEmbedding.mkString("[", ", ", "]")
        context.write(new Text(token.toString), new Text(embeddingStr))
        reducerLogger.debug(s"Wrote embedding for token $token")
      }
    }

    /**
     * Creates the neural network model for embedding generation.
     *
     * @return Initialized MultiLayerNetwork
     *
     * Design Rationale:
     * 1. Configurability: Uses external configuration for network parameters.
     * 2. Autoencoder Structure: Implements a simple autoencoder for embedding generation.
     * 3. Adam Optimizer: Efficient and widely used optimizer for neural networks.
     */
    private def createModel(): MultiLayerNetwork = {
      reducerLogger.info("Creating neural network model")
      val conf = new NeuralNetConfiguration.Builder()
        .updater(new Adam())
        .list()
        .layer(new DenseLayer.Builder().nIn(windowSize).nOut(embeddingSize)
          .activation(Activation.fromString(config.getString("neural-network.activation-function")))
          .build())
        .layer(new OutputLayer.Builder(LossFunctions.LossFunction.valueOf(config.getString("neural-network.loss-function")))
          .nIn(embeddingSize).nOut(windowSize)
          .activation(Activation.fromString(config.getString("neural-network.output-activation-function")))
          .build())
        .build()

      val network = new MultiLayerNetwork(conf)
      network.init()
      reducerLogger.info("Neural network model created and initialized")
      network
    }

    /**
     * Creates an input feature from a window of tokens.
     *
     * @param window Sequence of integer tokens
     * @return INDArray representation of the window
     *
     * Design Rationale:
     * 1. Conversion: Transforms integer tokens to float for neural network input.
     * 2. Reshaping: Ensures the input is in the correct shape for the network.
     */
    private def createInputFeature(window: Seq[Int]): INDArray = {
      val floatArray = window.map(_.toFloat).toArray
      val feature = Nd4j.create(floatArray).reshape(1, windowSize)
      reducerLogger.trace(s"Created input feature for window: ${window.mkString(",")}")
      feature
    }

    override def cleanup(context: Reducer[Text, Text, Text, Text]#Context): Unit = {
      reducerLogger.info("EmbeddingReducer cleanup completed")
    }
  }

  /**
   * Runs the embedding job.
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
    logger.info(s"Setting up Embedding job with input: $input and output: $output")

    val setupJob = Try(Job.getInstance(conf, jobName)).map { job =>
      job.setJarByClass(this.getClass)
      job.setMapperClass(classOf[EmbeddingMapper])
      job.setReducerClass(classOf[EmbeddingReducer])
      job.setMapOutputKeyClass(classOf[Text])
      job.setMapOutputValueClass(classOf[Text])
      job.setOutputKeyClass(classOf[Text])
      job.setOutputValueClass(classOf[Text])
      FileInputFormat.addInputPath(job, input)
      FileOutputFormat.setOutputPath(job, output)
      logger.debug("Job configuration completed")
      job
    }

    val cleanupOutput = setupJob.flatMap(_ => Try(FileSystem.get(conf))).map { fs =>
      if (fs.exists(output)) {
        logger.warn(s"Output path $output already exists. Deleting.")
        fs.delete(output, true)
      }
    }

    val runAndComplete = cleanupOutput.flatMap { _ =>
      Try {
        logger.info(s"Starting Embedding job execution")
        setupJob.get.waitForCompletion(true)
      }.flatMap { completed =>
        if (completed) {
          logger.info("Embedding job completed successfully")
          Success(())
        } else {
          logger.error("Embedding job failed")
          Failure(new Exception("Embedding job failed"))
        }
      }
    }

    runAndComplete.map { _ =>
      logger.info("Embedding job process finished")
    }
  }
}