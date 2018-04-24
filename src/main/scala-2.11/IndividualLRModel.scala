import org.apache.spark.ml.classification.{LogisticRegressionModel, LogisticRegressionTrainingSummary}
import org.apache.spark.ml.linalg
import org.apache.spark.sql.DataFrame

class IndividualLRModel(lrModel: LogisticRegressionModel,
                        testData: DataFrame,
                        reqColsInPrediction: Seq[String]) extends IndividualModel {

  val responseColName: String = lrModel.getLabelCol
  val learnerName = "LogisticRegression"

  type M = LogisticRegressionModel
  val model: LogisticRegressionModel = lrModel

  val predictionRelatedColumns: List[String] = List("prediction", "probability", "rawPrediction").map(responseColName ++ "_" ++ _)

  lazy val modelCoefficients: linalg.Vector = model.coefficients
  lazy val modelBias: Double = model.intercept

  lazy val trainSummary: LogisticRegressionTrainingSummary = model.summary
  lazy val trainPredictions: DataFrame = trainSummary.predictions

  lazy val testPredictions: DataFrame = model.transform(testData)

  def predict(df: DataFrame): DataFrame = model.transform(df)

}
