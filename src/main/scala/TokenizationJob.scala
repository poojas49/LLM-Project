import org.apache.hadoop.conf.Configuration
import org.apache.hadoop.fs.{FileSystem, Path}
import org.apache.hadoop.io.{IntWritable, LongWritable, Text}
import org.apache.hadoop.mapreduce.{Job, Mapper, Reducer}
import org.apache.hadoop.mapreduce.lib.input.FileInputFormat
import org.apache.hadoop.mapreduce.lib.output.FileOutputFormat
import com.knuddels.jtokkit.Encodings
import com.knuddels.jtokkit.api.{IntArrayList, ModelType}
import com.typesafe.config.ConfigFactory

import scala.jdk.CollectionConverters.*

object TokenizationJob {
  private val config = ConfigFactory.load()
  private val modelTypeString = config.getString("tokenization.model-type")
  private val jobName = config.getString("tokenization.job-name")
  private val preprocessingRegex = config.getString("tokenization.preprocessing-regex")

  private class TokenizationMapper extends Mapper[LongWritable, Text, Text, IntWritable] {
    private lazy val encoding = {
      val registry = Encodings.newDefaultEncodingRegistry()
      registry.getEncodingForModel(ModelType.valueOf(modelTypeString))
    }
    private val one = new IntWritable(1)

    private def preprocess(text: String): String = {
      text.toLowerCase
        .replaceAll(preprocessingRegex, "") // Remove all characters except lowercase letters, numbers, and spaces
        .split("\\s+") // Split by whitespace
        .mkString(" ") // Join back with single spaces
    }

    override def map(key: LongWritable, value: Text, context: Mapper[LongWritable, Text, Text, IntWritable]#Context): Unit = {
      val text = preprocess(value.toString)
      val words = text.split("\\s+") // Split by whitespace

      words.foreach { word =>
        val encodedTokens = encoding.encode(word)
        encodedTokens.toArray.foreach { token =>
          val tokenList = new IntArrayList(1)
          tokenList.add(token)
          val word = encoding.decode(tokenList)
          context.write(new Text(s"$word\t$token"), one)
        }
      }
    }
  }

  private class TokenizationReducer extends Reducer[Text, IntWritable, Text, Text] {
    override def reduce(key: Text, values: java.lang.Iterable[IntWritable], context: Reducer[Text, IntWritable, Text, Text]#Context): Unit = {
      val sum = values.asScala.map(_.get).sum
      val Array(word, token) = key.toString.split("\t")
      context.write(null, new Text(s"$word\t$token\t$sum"))
    }
  }

  def runJob(conf: Configuration, input: Path, output: Path): Unit = {
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
      fs.delete(output, true)
    }
    if (!job.waitForCompletion(true)) System.exit(1)
  }
}