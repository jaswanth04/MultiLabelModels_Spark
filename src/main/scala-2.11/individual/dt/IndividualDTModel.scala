package individual.dt

import individual.IndividualModel
import org.apache.spark.ml.classification.DecisionTreeClassificationModel
import org.apache.spark.sql.DataFrame

class IndividualDTModel(dtModel: DecisionTreeClassificationModel,
                        testData: DataFrame,
                        trainData: DataFrame,
                        reqColsInPrediction: Seq[String]) extends
  IndividualModel[DecisionTreeClassificationModel](model = dtModel,
    testData,
    trainData,
    reqCols = reqColsInPrediction) {

  val learnerName = "DecisionTree"

}
