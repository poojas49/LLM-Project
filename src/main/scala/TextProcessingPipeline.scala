import com.typesafe.config.ConfigFactory
import org.apache.hadoop.conf.Configuration
import org.apache.hadoop.fs.Path
import org.apache.hadoop.mapreduce.lib.input.FileInputFormat

import scala.util.{Failure, Success, Try}
import scala.jdk.CollectionConverters._

/**
 * TextProcessingPipeline object handles the execution of a series of Hadoop jobs
 * for processing text data. The pipeline is configurable through an external
 * configuration file, making it flexible and easy to maintain.
 *
 * Design Rationale:
 * 1. Configurability: External configuration allows for easy modification of the pipeline structure and parameters.
 * 2. Modularity: Each step in the pipeline is a separate Hadoop job, allowing for easy addition, removal, or modification of steps.
 * 3. Error Handling: Extensive use of Try monad for comprehensive error handling and propagation.
 * 4. Functional Approach: Utilizes functional programming concepts for cleaner, more maintainable code.
 * 5. Reflection: Uses reflection to dynamically load and execute job classes, providing flexibility in pipeline structure.
 *
 * Pipeline Overview:
 * 1. Tokenization Job: Splits input text into tokens.
 *    Input: Raw text files
 *    Output: Token-frequency pairs
 * 2. Sliding Window Job: Creates context windows for each token.
 *    Input: Token-frequency pairs
 *    Output: Window-label pairs
 * 3. Embedding Job: Generates vector embeddings for tokens.
 *    Input: Window-label pairs
 *    Output: Token-embedding pairs
 * 4. Semantic Similarity Job: Calculates similarity between token embeddings.
 *    Input: Token-embedding pairs
 *    Output: Token-similar_tokens pairs
 * 5. Statistics Collation Job: Combines results from previous jobs.
 *    Input: Results from Tokenization and Semantic Similarity jobs
 *    Output: YAML-formatted statistics for each token
 */
object TextProcessingPipeline {
  /**
   * Entry point of the application. Parses command-line arguments and initiates
   * the file processing.
   *
   * Design Rationale:
   * - Simple command-line interface for ease of use in various environments (e.g., cluster schedulers, scripts).
   * - Early exit with informative error message if arguments are incorrect, following fail-fast principle.
   *
   * @param args Command-line arguments: input path and output base path
   */
  def main(args: Array[String]): Unit = {
    args.toList match {
      case inputPath :: outputBasePath :: Nil =>
        processFiles(new Path(inputPath), new Path(outputBasePath))
      case _ =>
        System.err.println("Usage: TextProcessingPipeline <input path> <output path>")
        System.exit(-1)
    }
  }

  /**
   * Processes files by configuring Hadoop and running the pipeline.
   * Uses a functional approach with Try for error handling.
   *
   * Design Rationale:
   * - Separation of concerns: Configuration loading, Hadoop setup, and pipeline execution are separate steps.
   * - Use of for-comprehension for cleaner sequencing of operations that can fail.
   * - Comprehensive error handling with informative error messages for easier debugging.
   *
   * @param inputPath Input path for the pipeline (raw text files)
   * @param outputBasePath Base output path for all jobs in the pipeline
   */
  def processFiles(inputPath: Path, outputBasePath: Path): Unit = {
    val result = Try(ConfigFactory.load())
      .flatMap { config =>
        for {
          conf <- configureHadoop(config)
          output <- runPipeline(config, conf, inputPath, outputBasePath)
        } yield output
      }

    // Handle the result of the pipeline execution
    result match {
      case Success(output) => println(s"Pipeline completed. Final output: $output")
      case Failure(exception) =>
        System.err.println(s"Pipeline failed: ${exception.getMessage}")
        exception.printStackTrace()
    }
  }

  /**
   * Configures Hadoop settings based on the provided configuration.
   * This allows for easy adjustment of Hadoop parameters without changing the code.
   *
   * Design Rationale:
   * - Centralized configuration: All Hadoop-specific settings are in one place for easier management.
   * - Use of Try for error handling: Ensures that configuration errors are caught and propagated properly.
   * - Flexibility: Allows for easy addition or modification of Hadoop settings.
   *
   * @param config The loaded configuration
   * @return A Try[Configuration] with the Hadoop configuration
   */
  def configureHadoop(config: com.typesafe.config.Config): Try[Configuration] = Try {
    val conf = new Configuration()
    val dataShardSizeInMB = config.getLong("hadoop.split.size.mb")
    val numReducers = config.getInt("hadoop.num.reducers")
    conf.setLong(FileInputFormat.SPLIT_MAXSIZE, dataShardSizeInMB * 1024 * 1024)
    conf.setInt("mapreduce.job.reduces", numReducers)
    conf
  }

  /**
   * Runs the pipeline of Hadoop jobs as specified in the configuration.
   * This method demonstrates several key design principles:
   * 1. Flexibility: Jobs are dynamically loaded and executed based on the configuration.
   * 2. Error handling: Uses Try for composable error handling.
   * 3. Functional programming: Uses foldLeft for sequential job execution.
   *
   * Design Rationale:
   * - Dynamic job loading: Allows for easy modification of the pipeline structure through configuration.
   * - Use of reflection: Enables adding new job types without modifying the pipeline code.
   * - Sequential execution: Ensures that each job's output is available for the next job in the pipeline.
   * - Special handling of final job: Allows for different parameters or logic for the final aggregation step.
   *
   * @param config The loaded configuration
   * @param conf Hadoop configuration
   * @param inputPath Input path for the first job (raw text files)
   * @param outputBasePath Base output path for all jobs
   * @return A Try[Path] with the final output path (YAML-formatted statistics)
   */
  def runPipeline(config: com.typesafe.config.Config, conf: Configuration, inputPath: Path, outputBasePath: Path): Try[Path] = {
    // Load job configurations from the config file
    val jobsConfig = config.getConfigList("pipeline.jobs").asScala.toList
    val jobOutputs = jobsConfig.map { jobConfig =>
      val jobName = jobConfig.getString("name")
      val jobClass = jobConfig.getString("class")
      // Use reflection to get the runJob method of each job class
      (jobName, Class.forName(jobClass).getMethod("runJob", classOf[Configuration], classOf[Path], classOf[Path]))
    }

    // Each job's output becomes the input for the next job
    val pipelineResult = jobOutputs.foldLeft(Try(inputPath)) {
      case (prevOutputTry, (jobName, runJobMethod)) =>
        prevOutputTry.flatMap { prevOutput =>
          val output = new Path(outputBasePath, jobName)
          // Invoke the runJob method and return the output path on success
          Try(runJobMethod.invoke(null, conf, prevOutput, output)).map(_ => output)
        }
    }

    // Run the final statistics collation job
    pipelineResult.flatMap { _ =>
      val tokenizationOutput = new Path(outputBasePath, jobsConfig.head.getString("name"))
      val semanticSimilarityOutput = new Path(outputBasePath, jobsConfig.last.getString("name"))
      val finalJobConfig = config.getConfig("pipeline.final-job")
      val finalOutput = new Path(outputBasePath, finalJobConfig.getString("name"))
      conf.setInt("mapreduce.job.reduces", finalJobConfig.getInt("num.reducers"))
      Try(StatisticsCollaterJob.runJob(conf, tokenizationOutput, semanticSimilarityOutput, finalOutput))
        .map(_ => finalOutput)
    }
  }
}

/**
 * Note: The following job classes are not defined in this file but are referenced in the pipeline:
 *
 * TokenizationJob:
 * - Input: Raw text files
 * - Output: Token-frequency pairs (word, token, frequency)
 * - Mapper: Splits text into words, tokenizes each word
 * - Reducer: Aggregates frequencies for each token
 *
 * SlidingWindowJob:
 * - Input: Token-frequency pairs
 * - Output: Window-label pairs (input window, next token as label)
 * - Mapper: Emits individual tokens
 * - Reducer: Creates sliding windows and emits window-label pairs
 *
 * EmbeddingJob:
 * - Input: Window-label pairs
 * - Output: Token-embedding pairs
 * - Mapper: Prepares input for the neural network
 * - Reducer: Runs the neural network to generate embeddings
 *
 * SemanticSimilarityJob:
 * - Input: Token-embedding pairs
 * - Output: Token-similar_tokens pairs
 * - Mapper: Emits token-embedding pairs
 * - Reducer: Calculates cosine similarity between embeddings
 *
 * StatisticsCollaterJob:
 * - Input: Results from Tokenization and Semantic Similarity jobs
 * - Output: YAML-formatted statistics for each token
 * - Mapper: Processes input from both sources
 * - Reducer: Combines statistics and formats output as YAML
 */