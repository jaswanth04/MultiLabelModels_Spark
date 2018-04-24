import org.apache.spark.sql.DataFrame

abstract class IndividualModel {

  type M
  def learnerName: String
  def model: M
  def trainPredictions: DataFrame
  def testPredictions: DataFrame
  def predict: DataFrame

}
