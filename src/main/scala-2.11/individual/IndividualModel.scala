package individual

import evaluation.{ConfusionMatrix, IndividualModelEvaluation, IndividualModelMetrics}
import org.apache.spark.ml.linalg.Vector
import org.apache.spark.ml.classification.ClassificationModel
import org.apache.spark.sql.DataFrame

abstract class IndividualModel[M <: ClassificationModel[Vector, M]](model: M,
                                                         testData: DataFrame,
                                                         trainData: DataFrame,
                                                         reqCols: Seq[String]) extends IndividualModelEvaluation {

  def learnerName: String
  val indModel: M = model
  val responseColName: String = indModel.getLabelCol
  def predict(df: DataFrame): DataFrame = indModel.transform(df)
  def trainPredictions: DataFrame = this.predict(trainData)
  def testPredictions: DataFrame = this.predict(testData)

  def trainConfusionMatrix: ConfusionMatrix = evaluate(df = trainPredictions,
    actualColumnName = responseColName,
    predictedColumnName = responseColName.concat("_prediction"))
  def testConfusionMatrix: ConfusionMatrix = evaluate(df = testPredictions,
    actualColumnName = responseColName,
    predictedColumnName = responseColName.concat("_prediction"))

  def trainMetrics: IndividualModelMetrics = getIndividualModelMetrics(df = trainPredictions,
    actualColumnName = responseColName,
    predictedColumnName = responseColName.concat("_prediction"))
  def testMetrics: IndividualModelMetrics = getIndividualModelMetrics(df = testPredictions,
    actualColumnName = responseColName,
    predictedColumnName = responseColName.concat("_prediction"))
}
