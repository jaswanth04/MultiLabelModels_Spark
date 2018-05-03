package individual.dt

import individual.IndividualModelBuilder
import org.apache.spark.ml.classification.{DecisionTreeClassifier, DecisionTreeClassificationModel}
import org.apache.spark.sql.DataFrame

class IndividualDTModelBuilder(trainDf: DataFrame,
                               vectorizedFeatureColName: String,
                               responseColName: String,
                               reqColsInPredictions: Seq[String] = Seq(),
                               impurity: String = "gini",
                               maxBins: Int = 32,
                               maxDepth: Int = 5,
                               minInfoGain: Double = 0.0,
                               minInstancesPerNode : Int = 1,
                               testDf:DataFrame) extends
  IndividualModelBuilder[DecisionTreeClassificationModel, DecisionTreeClassifier](featureColName = vectorizedFeatureColName,
    trainDf = trainDf,
    testDf = testDf,
    responseColName = responseColName) {

  val learnerName = "DecisionTree"
  override val reqCols: Seq[String] = reqColsInPredictions

  println(s"Building model for the feature: $vectorizedFeatureColumnName")

//  type L = DecisionTreeClassifier


  def learner: DecisionTreeClassifier = new DecisionTreeClassifier()
    .setImpurity(impurity)
    .setMaxBins(maxBins)
    .setMaxDepth(maxDepth)
    .setMinInfoGain(minInfoGain)
    .setMinInstancesPerNode(minInstancesPerNode)
    .setFeaturesCol(vectorizedFeatureColName)
    .setLabelCol(responseColumnName)
    .setPredictionCol(predictionCols.head)
    .setProbabilityCol(predictionCols(1))
    .setRawPredictionCol(predictionCols(2))

  def buildModel: IndividualDTModel = new IndividualDTModel(dtModel = learner.fit(trainData),
    testData = testData,
    trainData = trainData,
    reqColsInPrediction = this.reqColsInPredictions)

}
