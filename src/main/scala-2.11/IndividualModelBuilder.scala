import org.apache.spark.sql.DataFrame
import org.apache.spark.sql.functions.col

abstract class IndividualModelBuilder {

  type L
  def learnerName: String
  def learner: L
  def buildModel: IndividualModel
  def vectorizedFeatureColumnName: String
  def responseColumnName: String

  def getRequiredColumns(df: DataFrame,
                         requiredColumnNames: Seq[String]): DataFrame = {
    val columnNames = requiredColumnNames ++ List(vectorizedFeatureColumnName, responseColumnName)
    df.select(columnNames.map(col): _*)
  }

}
