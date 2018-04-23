import org.apache.spark.sql.DataFrame

abstract class dataLoader {

  def dataDf: DataFrame
  def getFinalDf: DataFrame
}
