package IndividualModels.LR

import IndividualModels.IndividualModelBuilder
import org.apache.spark.ml.classification.LogisticRegression
import org.apache.spark.sql.DataFrame

class IndividualLRModelBuilder(trainData: DataFrame,
                               vectorizedFeatureColName: String,
                               responseColName: String,
                               reqColsInPredictions: Seq[String] = Seq(),
                               maxIterations: Int = 100,
                               regularizationParameter: Double = 0.0,
                               elasticNetParameter: Double = 0.0,
                               testData:DataFrame) extends IndividualModelBuilder {

  val responseColumnName: String = responseColName
  val predictionCols: List[String] = List("prediction", "probability", "rawPrediction").map(responseColumnName ++ "_" ++ _)
  val vectorizedFeatureColumnName: String = vectorizedFeatureColName
  val learnerName = "LogisticRegression"

  println(s"Building model for the feature: $vectorizedFeatureColumnName")

  type L = LogisticRegression

  def learner: LogisticRegression = new LogisticRegression()
    .setFeaturesCol(vectorizedFeatureColumnName)
    .setLabelCol(responseColumnName)
    .setMaxIter(maxIterations)
    .setRegParam(regularizationParameter)
    .setElasticNetParam(elasticNetParameter)
    .setTol(1e-9)
    .setPredictionCol(predictionCols.head)
    .setProbabilityCol(predictionCols(1))
    .setRawPredictionCol(predictionCols(2))

  def buildModel: IndividualLRModel = new IndividualLRModel(lrModel = learner.fit(trainData),
    testData = testData,
    reqColsInPrediction = this.reqColsInPredictions)

}
