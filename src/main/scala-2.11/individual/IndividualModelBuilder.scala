package individual

import org.apache.spark.ml.classification.{ClassificationModel, Classifier}
import org.apache.spark.sql.DataFrame
import org.apache.spark.sql.functions.col
import org.apache.spark.ml.linalg.Vector


abstract class IndividualModelBuilder[M <: ClassificationModel[Vector, M] ,L <: Classifier[Vector, L, M]](featureColName: String,
                                                                                                          responseColName:String,
                                                                                                          trainDf: DataFrame,
                                                                                                          testDf: DataFrame) {

  def learnerName: String
  val trainData: DataFrame = trainDf
  val testData: DataFrame = testDf
  val reqCols: Seq[String]
  def learner: L
  def buildModel: IndividualModel[M]
  def vectorizedFeatureColumnName: String = featureColName
  def responseColumnName: String = responseColName
  val predictionCols: List[String] = List("prediction", "probability", "rawPrediction").map(responseColumnName ++ "_" ++ _)

  def getRequiredColumns(df: DataFrame,
                         requiredColumnNames: Seq[String]): DataFrame = {
    val columnNames = requiredColumnNames ++ List(vectorizedFeatureColumnName, responseColumnName)
    df.select(columnNames.map(col): _*)
  }

}
