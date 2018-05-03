import dataloader.FullTableData
import multilabel.dt.MultiLabelDTModelBuilder
import multilabel.lr.MultiLabelLRModelBuilder
import org.apache.log4j.{Level, Logger}
import org.apache.spark.sql.{DataFrame, SparkSession}

object multiLabel_trial {

  def main(args: Array[String]): Unit = {

    val spark = SparkSession.builder().master("local").getOrCreate()
    val rootLogger = Logger.getRootLogger
    rootLogger.setLevel(Level.ERROR)

    val ftData = new FullTableData("G:\\MultiLabelModels_Spark\\data\\niceDataJoinedFinal_1.csv", spark)
    val dataDf = ftData.getFinalDf

    val y_flags = Seq("flag27", "flag16", "flag39")
    val x_flags = dataDf.schema.fieldNames.filter(_ startsWith "x_flag").toList

    def splitTrainTest(df: DataFrame): (DataFrame, DataFrame) = (df.filter(df("filedt") === 201701), df.filter(df("filedt") === 201702))
/*
    // Multi Label LR Model
    val mlModelMaker = new MultiLabelLRModelBuilder(data = dataDf,
      xFlagNames = x_flags,
      responseColumns = y_flags,
      trainTestSplitFunc = splitTrainTest,
      reqColsInPrediction = List("vcaccountnumber","filedt"))
*/

    // MultiLabel DT Model

    val mlModelMaker = new MultiLabelDTModelBuilder(data = dataDf,
      xFlagNames = x_flags,
      responseColumns = y_flags,
      trainTestSplitFunc = splitTrainTest,
      reqColsInPrediction = List("vcaccountnumber","filedt"))
    val mlModel = mlModelMaker.buildMultiLabelModels("CC")

    mlModel.trainPredictions.show()
    mlModel.testPredictions.show()

    println(mlModel.testPerformance)
    println(mlModel.trainPerformance)

  }
}
