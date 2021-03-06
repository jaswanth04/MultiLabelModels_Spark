package individual.lr

import individual.IndividualModel
import org.apache.spark.ml.classification.LogisticRegressionModel
import org.apache.spark.ml.linalg
import org.apache.spark.sql.DataFrame

class IndividualLRModel(lrModel: LogisticRegressionModel,
                        testData: DataFrame,
                        trainData: DataFrame,
                        reqColsInPrediction: Seq[String]) extends
  IndividualModel[LogisticRegressionModel](model = lrModel,
    testData,
    trainData,
    reqCols = reqColsInPrediction) {

  val learnerName = "LogisticRegression"

  lazy val modelCoefficients: linalg.Vector = indModel.coefficients
  lazy val modelBias: Double = indModel.intercept

}
