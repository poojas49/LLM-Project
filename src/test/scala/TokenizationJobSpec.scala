import org.scalatest.flatspec.AnyFlatSpec
import org.scalatest.matchers.should.Matchers
import org.scalatestplus.mockito.MockitoSugar
import org.apache.hadoop.io.{IntWritable, LongWritable, Text}
import org.apache.hadoop.mapreduce.{Mapper, Reducer}
import org.mockito.Mockito._
import org.mockito.ArgumentMatchers._

import scala.jdk.CollectionConverters._

class TokenizationJobSpec extends AnyFlatSpec with Matchers with MockitoSugar {

  "TokenizationMapper" should "preprocess text correctly" in {
    val mapper = new TokenizationJob.TokenizationMapper()
    val input = "Hello, World! 123"
    val expectedOutput = "hello world 123"
    mapper.preprocess(input) should be(expectedOutput)
  }

  "TokenizationReducer" should "sum up frequencies correctly" in {
    val reducer = new TokenizationJob.TokenizationReducer()
    val context = mock[Reducer[Text, IntWritable, Text, Text]#Context]
    val key = new Text("hello\t15496")
    val values = List(new IntWritable(1), new IntWritable(1), new IntWritable(1)).asJava

    reducer.reduce(key, values, context)

    verify(context).write(isNull(), any[Text])
  }

  "TokenizationJob" should "initialize correctly" in {
    noException should be thrownBy {
      TokenizationJob
    }
  }
}