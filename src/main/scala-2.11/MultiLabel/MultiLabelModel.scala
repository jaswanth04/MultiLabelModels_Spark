package MultiLabel

import Individual.IndividualModel
import org.apache.spark.ml.classification.ClassificationModel
import org.apache.spark.sql.DataFrame
import org.apache.spark.ml.linalg.Vector

abstract class MultiLabelModel[M <: ClassificationModel[Vector, M], +IM <: IndividualModel[M]](modelList: List[IndividualModel[M]],
                                                                                    trainDf: DataFrame,
                                                                                    testDf: DataFrame,
                                                                                    reqCols: Seq[String]) {

  def learnerName: String
  val models: List[IndividualModel[M]] = modelList
  val responseCols: List[String] = models.map(_.responseColName)
  def predict(df: DataFrame): DataFrame = {
    def _pre(df: DataFrame, models: List[IndividualModel[M]]): DataFrame = {
      models match {
        case List() => df
        case head :: tail => _pre(head.predict(df), tail)
      }
    }
    _pre(df,models)
  }
  def trainPredictions: DataFrame = this.predict(trainDf)
  def testPredictions: DataFrame = this.predict(testDf)

}
