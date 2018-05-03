package multilabel.lr

import individual.lr.IndividualLRModel
import multilabel.MultiLabelModel
import org.apache.spark.ml.classification.LogisticRegressionModel
import org.apache.spark.sql.DataFrame

class MultiLabelLRModel(modelList: List[IndividualLRModel],
                        trainDf: DataFrame,
                        modelType: String,
                        testDf: DataFrame,
                        reqCols: Seq[String]) extends MultiLabelModel[LogisticRegressionModel,
  IndividualLRModel](modelList = modelList,
  modelAlgorithm = modelType,
  trainDf = trainDf,
  testDf = testDf,
  reqCols = reqCols)  {

  val learnerName = "LogisticRegression"

}
