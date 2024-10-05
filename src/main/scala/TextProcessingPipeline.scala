import org.apache.hadoop.conf.Configuration
import org.apache.hadoop.fs.Path

object TextProcessingPipeline {
  def main(args: Array[String]): Unit = {
    if (args.length != 2) {
      System.err.println("Usage: TextProcessingPipeline <input path> <output path>")
      System.exit(-1)
    }

    TextShardingJob.runJob(args(0))

    val inputPath = new Path(args(0))
    val outputBasePath = new Path(args(1))
    val conf = new Configuration()

    val jobOutputs = List(
      ("tokenization", TokenizationJob.runJob),
      ("sliding_window", SlidingWindowJob.runJob),
      ("embedding", EmbeddingJob.runJob),
      ("semantic_similarity", SemanticSimilarityJob.runJob)
    )

    jobOutputs.foldLeft(inputPath) { case (prevOutput, (jobName, runJob)) =>
      val output = new Path(outputBasePath, jobName)
      runJob(conf, prevOutput, output)
      output
    }

    // Run the Statistics Collater Job
    val tokenizationOutput = new Path(outputBasePath, "tokenization")
    val semanticSimilarityOutput = new Path(outputBasePath, "semantic_similarity")
    val finalOutput = new Path(outputBasePath, "statistics_collated")
    StatisticsCollaterJob.runJob(conf, tokenizationOutput, semanticSimilarityOutput, finalOutput)

    println(s"Pipeline completed. Final output: $finalOutput")
  }
}