import dataloader.FullTableData
import org.apache.spark.sql.{DataFrame, SparkSession}
import org.apache.log4j.{Level, Logger}
import org.apache.spark.ml.feature.VectorAssembler

object IndModelTrial {

  def main(args: Array[String]): Unit = {
    val spark = SparkSession.builder().master("local").getOrCreate()
    val rootLogger = Logger.getRootLogger
    rootLogger.setLevel(Level.ERROR)

    val ftData = new FullTableData("G:\\MultiLabelModels_Spark\\data\\niceDataJoinedFinal_1.csv", spark)
    val dataDf = ftData.getFinalDf

    val y_flag = "flag27"
    val x_flags = dataDf.schema.fieldNames.filter(_ startsWith "x_flag").toList

    val vecAss = new VectorAssembler().setInputCols(x_flags.toArray).setOutputCol("features")
    val vectorizedDf: DataFrame = vecAss.transform(dataDf).select("features", "flag27", "filedt")

    def splitTrainTest(df: DataFrame): (DataFrame, DataFrame) = (df.filter(df("filedt") === 201701), df.filter(df("filedt") === 201702))

    val (trainDf, testDf) = splitTrainTest(vectorizedDf)

    val iModel = new individual.lr.IndividualLRModelBuilder(trainDf = trainDf,
      vectorizedFeatureColName = "features",
      responseColName = y_flag,
      testDf = testDf)

    val fiModel = iModel.buildModel

    fiModel.trainPredictions.show()
    fiModel.testPredictions.show()

    println(fiModel.trainMetrics)
    println(fiModel.testMetrics)
  }

}
