import org.apache.spark.{SparkConf, SparkContext}

object trial {

  def main(args: Array[String]): Unit = {
    val conf = new SparkConf().setMaster("local").setAppName("trial")
    val sc = new SparkContext(conf)

    val data = sc.parallelize(List(1, 2, 3, 4, 5, 6))

    data.map(_ + 2).foreach(println)
  }
}
