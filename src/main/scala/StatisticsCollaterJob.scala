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

object StatisticsCollaterJob {
  private val config = ConfigFactory.load()
  private val jobName = config.getString("statistics-collater.job-name")
  private val inputSplitDelimiter = config.getString("statistics-collater.input-split-delimiter")
  private val similarityPairDelimiter = config.getString("statistics-collater.similarity-pair-delimiter")
  private val similarityScoreDelimiter = config.getString("statistics-collater.similarity-score-delimiter")
  private val naValue = config.getString("statistics-collater.na-value")

  private val wordKey = config.getString("yaml-output.word-key")
  private val intTokenKey = config.getString("yaml-output.int-token-key")
  private val frequencyKey = config.getString("yaml-output.frequency-key")
  private val similarTokensKey = config.getString("yaml-output.similar-tokens-key")

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

  class StatisticsReducer extends Reducer[Text, Text, Text, Text] {
    override def reduce(key: Text, values: java.lang.Iterable[Text], context: Reducer[Text, Text, Text, Text]#Context): Unit = {
      val token = key.toString

      val (word, frequency, similarities) = values.asScala.foldLeft(("", 0L, List.empty[String])) {
        case ((word, frequency, similarities), value) =>
          val parts = value.toString.split(inputSplitDelimiter)
          if (parts.length == 2) {
            (parts(0), parts(1).toLong, similarities)
          } else {
            val similarityPairs = value.toString.split(similarityPairDelimiter).map { pair =>
              val Array(similarToken, score) = pair.split(s"\\$similarityScoreDelimiter")
              s"$similarToken ($score)"
            }.toList
            (word, frequency, similarityPairs)
          }
      }

      val data = Map(
        wordKey -> word,
        intTokenKey -> token,
        frequencyKey -> frequency,
        similarTokensKey -> (if (similarities.isEmpty) naValue else similarities.asJava)
      )

      val yaml = new Yaml()
      val writer = new StringWriter()
      yaml.dump(data.asJava, writer)

      context.write(null, new Text(writer.toString))
    }
  }

  def runJob(conf: Configuration, tokenizationInput: Path, similarityInput: Path, output: Path): Unit = {
    val job = Job.getInstance(conf, jobName)
    job.setJarByClass(this.getClass)
    job.setReducerClass(classOf[StatisticsReducer])
    job.setOutputKeyClass(classOf[Text])
    job.setOutputValueClass(classOf[Text])

    MultipleInputs.addInputPath(job, tokenizationInput, classOf[TextInputFormat], classOf[TokenizationMapper])
    MultipleInputs.addInputPath(job, similarityInput, classOf[TextInputFormat], classOf[SimilarityMapper])

    FileOutputFormat.setOutputPath(job, output)

    val fs = FileSystem.get(conf)
    if (fs.exists(output)) {
      fs.delete(output, true)
    }

    if (!job.waitForCompletion(true)) System.exit(1)
  }
}