package individual.lsvc

import individual.IndividualModel
import org.apache.spark.ml.classification.LinearSVCModel
import org.apache.spark.sql.DataFrame

class IndividualLSVCModel(lsvcModel: LinearSVCModel,
                          testData: DataFrame,
                          trainData: DataFrame,
                          reqColsInPrediction: Seq[String]) extends
  IndividualModel[LinearSVCModel](model = lsvcModel,
    testData,
    trainData,
    reqCols = reqColsInPrediction) {

  val learnerName = "LinearSupportVectorClassification"

}
