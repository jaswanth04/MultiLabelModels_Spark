package multilabel

import individual.IndividualModel
import evaluation.{MultiLabelMetrics, MultiLabelModelEvaluation}
import org.apache.spark.ml.classification.ClassificationModel
import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.sql.DataFrame
import org.apache.spark.ml.linalg.Vector

abstract class MultiLabelModel[M <: ClassificationModel[Vector, M], +IM <: IndividualModel[M]](modelList: List[IndividualModel[M]],
                                                                                               modelAlgorithm: String,
                                                                                    trainDf: DataFrame,
                                                                                    testDf: DataFrame,
                                                                                    reqCols: Seq[String]) extends MultiLabelModelEvaluation {

  def learnerName: String
  val models: List[IndividualModel[M]] = modelList
  val responseCols: List[String] = models.map(_.responseColName)
  val modelType: String = modelAlgorithm
  def predict(df: DataFrame): DataFrame =  {
    def _pre(df: DataFrame, models: List[IndividualModel[M]]): DataFrame = {
      models match {
        case List() => df
        case head :: tail => if (modelType == "BR") _pre(head.predict(df), tail)
        else {
          val predictedDf = head.predict(df)
          val vecAss = new VectorAssembler()
            .setInputCols(Array(head.featureName, head.responseColName ++ "_prediction"))
            .setOutputCol("feature_" ++ head.responseColName)
          _pre(vecAss.transform(predictedDf).drop(head.featureName).persist(), tail)
        }
      }
    }
    _pre(df,models)
  }
  def trainPredictions: DataFrame = this.predict(trainDf)
  def testPredictions: DataFrame = this.predict(testDf)

  def trainPerformance: MultiLabelMetrics = getEvaluationMetrics(trainPredictions, this.responseCols)
  def testPerformance: MultiLabelMetrics = getEvaluationMetrics(testPredictions, this.responseCols)

}
