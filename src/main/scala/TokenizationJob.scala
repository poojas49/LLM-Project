import org.apache.hadoop.conf.Configuration
import org.apache.hadoop.fs.{FileSystem, Path}
import org.apache.hadoop.io.{IntWritable, LongWritable, Text}
import org.apache.hadoop.mapreduce.{Job, Mapper, Reducer}
import org.apache.hadoop.mapreduce.lib.input.FileInputFormat
import org.apache.hadoop.mapreduce.lib.output.FileOutputFormat
import com.knuddels.jtokkit.Encodings
import com.knuddels.jtokkit.api.{IntArrayList, ModelType}

import scala.jdk.CollectionConverters.*

object TokenizationJob {
  private class TokenizationMapper extends Mapper[LongWritable, Text, Text, IntWritable] {
    private lazy val encoding = {
      val registry = Encodings.newDefaultEncodingRegistry()
      registry.getEncodingForModel(ModelType.TEXT_EMBEDDING_ADA_002)
    }
    private val one = new IntWritable(1)

    private def preprocess(text: String): String = {
      text.toLowerCase
        .replaceAll("[^a-z0-9\\s]", "") // Remove all characters except lowercase letters, numbers, and spaces
        .split("\\s+") // Split by whitespace
        .mkString(" ") // Join back with single spaces
    }

    override def map(key: LongWritable, value: Text, context: Mapper[LongWritable, Text, Text, IntWritable]#Context): Unit = {
      val text = value.toString.toLowerCase.replaceAll("[^a-z0-9\\s]", "") // Remove all characters except lowercase letters, numbers, and spaces
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
    val job = Job.getInstance(conf, "JTokkit Tokenization")
    job.setJarByClass(this.getClass)
    job.setMapperClass(classOf[TokenizationMapper])
    job.setReducerClass(classOf[TokenizationReducer])
    job.setOutputKeyClass(classOf[Text])
    job.setOutputValueClass(classOf[IntWritable])
    FileInputFormat.addInputPath(job, input)
    FileOutputFormat.setOutputPath(job, output)

//  Delete output path if it exists
    val fs = FileSystem.get(conf)
    if (fs.exists(output)) {
      fs.delete(output, true)
    }
    if (!job.waitForCompletion(true)) System.exit(1)
  }
}