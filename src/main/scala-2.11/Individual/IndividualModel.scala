package Individual

//import org.apache.spark.ml.classification.ClassificationModel
import org.apache.spark.ml.linalg.Vector
import org.apache.spark.ml.classification.ClassificationModel
import org.apache.spark.sql.DataFrame

abstract class IndividualModel[M <: ClassificationModel[Vector, M]](model: M,
                                                         testData: DataFrame,
                                                         trainData: DataFrame,
                                                         reqCols: Seq[String]) {

  def learnerName: String
  val indModel: M = model
  val responseColName: String = indModel.getLabelCol
  def predict(df: DataFrame): DataFrame = indModel.transform(df)
  def trainPredictions: DataFrame = this.predict(trainData)
  def testPredictions: DataFrame = this.predict(testData)


}
