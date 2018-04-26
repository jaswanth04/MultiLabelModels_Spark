package MultiLabel

import Individual.IndividualModel
import org.apache.spark.sql.DataFrame
import org.apache.spark.sql.functions.col

abstract class MultiLabelModel[M, IM <: IndividualModel[M]](modelList: List[IndividualModel[M]],
                                                             trainDf: DataFrame,
                                                             testDf: DataFrame,
                                                             reqCols: Seq[String]) {

  def learnerName: String
  val models: List[IndividualModel[M]] = modelList
  val responseCols: List[String] = models.map(_.responseColName)
  def predict(df: DataFrame): DataFrame = models.map(_.predict(df)).reduceLeft((df1, df2) => df1.join(df2, col("column").equalTo(df2("column"))))
  def trainPredictions: DataFrame = this.predict(trainDf)
  def testPredictions: DataFrame = this.predict(testDf)

}
