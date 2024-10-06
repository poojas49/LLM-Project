import org.scalatest.flatspec.AnyFlatSpec
import org.scalatest.matchers.should.Matchers
import org.scalatestplus.mockito.MockitoSugar
import org.apache.hadoop.io.{LongWritable, Text, IntWritable}
import org.apache.hadoop.mapreduce.{Mapper, Reducer}
import org.mockito.Mockito._
import org.mockito.ArgumentMatchers._
import com.typesafe.config.ConfigFactory

import scala.jdk.CollectionConverters._

class SlidingWindowJobSpec extends AnyFlatSpec with Matchers with MockitoSugar {

  "SlidingWindowJob" should "initialize with correct configuration" in {
    noException should be thrownBy {
      SlidingWindowJob
    }
  }

  "SlidingWindowMapper" should "extract and emit tokens correctly" in {
    val mapper = new SlidingWindowJob.SlidingWindowMapper()
    val context = mock[Mapper[LongWritable, Text, Text, Text]#Context]

    val inputLine = "word\t12345\t10"
    mapper.map(new LongWritable(0), new Text(inputLine), context)

    verify(context).write(new Text("data"), new Text("12345"))
  }

  it should "handle input lines with insufficient parts" in {
    val mapper = new SlidingWindowJob.SlidingWindowMapper()
    val context = mock[Mapper[LongWritable, Text, Text, Text]#Context]

    val inputLine = "insufficient"
    mapper.map(new LongWritable(0), new Text(inputLine), context)

    verify(context, never()).write(any[Text], any[Text])
  }

  "SlidingWindowReducer" should "process tokens using sliding window approach" in {
    val reducer = new SlidingWindowJob.SlidingWindowReducer()
    val context = mock[Reducer[Text, Text, Text, Text]#Context]

    val tokens = List("token1", "token2", "token3", "token4", "token5").map(new Text(_)).asJava
    reducer.reduce(new Text("data"), tokens, context)

    verify(context).write(new Text("input_0"), new Text("token1,token2,token3,token4"))
    verify(context).write(new Text("label_0"), new Text("token5"))
  }

  it should "handle cases with insufficient tokens for a full window" in {
    val reducer = new SlidingWindowJob.SlidingWindowReducer()
    val context = mock[Reducer[Text, Text, Text, Text]#Context]

    val tokens = List("token1", "token2").map(new Text(_)).asJava
    reducer.reduce(new Text("data"), tokens, context)

    verify(context, never()).write(any[Text], any[Text])
  }
}