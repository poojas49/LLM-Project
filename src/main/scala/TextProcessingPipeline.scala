import com.typesafe.config.ConfigFactory
import org.apache.hadoop.conf.Configuration
import org.apache.hadoop.fs.Path
import org.apache.hadoop.mapreduce.lib.input.FileInputFormat

import scala.util.{Failure, Success, Try}

object TextProcessingPipeline {
  def main(args: Array[String]): Unit = {
    args.toList match {
      case inputPath :: outputBasePath :: Nil =>
        processFiles(new Path(inputPath), new Path(outputBasePath))
      case _ =>
        System.err.println("Usage: TextProcessingPipeline <input path> <output path>")
        System.exit(-1)
    }
  }

  def processFiles(inputPath: Path, outputBasePath: Path): Unit = {
    val result = Try(ConfigFactory.load())
      .map { config =>
        (
          config.getLong("hadoop.split.size.mb"),
          config.getInt("hadoop.num.reducers"),
        )
      }
      .map { case (dataShardSizeInMB, numReducers) =>
        val conf = new Configuration()
        conf.setLong(FileInputFormat.SPLIT_MAXSIZE, dataShardSizeInMB * 1024 * 1024)
        conf.setInt("mapreduce.job.reduces", numReducers)
        conf
      }
      .flatMap(conf => runPipeline(conf, inputPath, outputBasePath))

    result match {
      case Success(output) => println(s"Pipeline completed. Final output: $output")
      case Failure(exception) =>
        System.err.println(s"Pipeline failed: ${exception.getMessage}")
        exception.printStackTrace()
    }
  }

  def runPipeline(conf: Configuration, inputPath: Path, outputBasePath: Path): Try[Path] = {
    val jobOutputs = List(
      ("tokenization", TokenizationJob.runJob),
      ("sliding_window", SlidingWindowJob.runJob),
      ("embedding", EmbeddingJob.runJob),
      ("semantic_similarity", SemanticSimilarityJob.runJob)
    )

    val pipelineResult = jobOutputs.foldLeft(Try(inputPath)) {
      case (prevOutputTry, (jobName, runJob)) =>
        prevOutputTry.flatMap { prevOutput =>
          val output = new Path(outputBasePath, jobName)
          Try(runJob(conf, prevOutput, output)).map(_ => output)
        }
    }

    pipelineResult.flatMap { _ =>
      val tokenizationOutput = new Path(outputBasePath, "tokenization")
      val semanticSimilarityOutput = new Path(outputBasePath, "semantic_similarity")
      val finalOutput = new Path(outputBasePath, "statistics_collated")
      conf.setInt("mapreduce.job.reduces", 1)
      Try(StatisticsCollaterJob.runJob(conf, tokenizationOutput, semanticSimilarityOutput, finalOutput))
        .map(_ => finalOutput)
    }
  }
}