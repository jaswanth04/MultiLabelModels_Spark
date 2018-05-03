package multilabel.lsvc

import individual.lsvc.IndividualLSVCModel
import multilabel.MultiLabelModel
import org.apache.spark.ml.classification.LinearSVCModel
import org.apache.spark.sql.DataFrame

class MultiLabelLSVCModel(modelList: List[IndividualLSVCModel],
                          trainDf: DataFrame,
                          modelType: String,
                          testDf: DataFrame,
                          reqCols: Seq[String]) extends MultiLabelModel[LinearSVCModel,
  IndividualLSVCModel](modelList = modelList,
  modelAlgorithm = modelType,
  trainDf = trainDf,
  testDf = testDf,
  reqCols = reqCols)   {

  val learnerName = "LinearSupportVectorClassifier"

}
