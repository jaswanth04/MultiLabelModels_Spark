package multilabel.dt

import individual.dt.IndividualDTModel
import multilabel.MultiLabelModel
import org.apache.spark.ml.classification.DecisionTreeClassificationModel
import org.apache.spark.sql.DataFrame

class MultiLabelDTModel(modelList: List[IndividualDTModel],
                        trainDf: DataFrame,
                        modelType: String,
                        testDf: DataFrame,
                        reqCols: Seq[String]) extends MultiLabelModel[DecisionTreeClassificationModel,
  IndividualDTModel](modelList = modelList,
  modelAlgorithm = modelType,
  trainDf = trainDf,
  testDf = testDf,
  reqCols = reqCols)   {

  val learnerName = "DecisionTree"

}
