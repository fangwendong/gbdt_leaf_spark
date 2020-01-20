/*
 * Licensed to the Apache Software Foundation (ASF) under one or more
 * contributor license agreements.  See the NOTICE file distributed with
 * this work for additional information regarding copyright ownership.
 * The ASF licenses this file to You under the Apache License, Version 2.0
 * (the "License"); you may not use this file except in compliance with
 * the License.  You may obtain a copy of the License at
 *
 *    http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package org.apache.spark.ml.mleap.gbdt

import com.github.fommil.netlib.BLAS.{getInstance => blas}
import org.json4s.{DefaultFormats, JObject}
import org.json4s.JsonDSL._
import org.apache.spark.annotation.{DeveloperApi, Since}
import org.apache.spark.internal.Logging
import org.apache.spark.ml.attribute.{Attribute, AttributeGroup, NumericAttribute}
import org.apache.spark.ml.classification.{ProbabilisticClassificationModel, ProbabilisticClassifier}
import org.apache.spark.ml.feature.LabeledPoint
import org.apache.spark.ml.linalg.{DenseVector, SparseVector, Vector, VectorUDT, Vectors}
import org.apache.spark.ml.param.shared.HasWeightCol
import org.apache.spark.ml.param.{Param, ParamMap}
import org.apache.spark.ml.regression.DecisionTreeRegressionModel
import org.apache.spark.ml.tree.Node
import org.apache.spark.mllib.tree.model.{Node => MlibNode}
import org.apache.spark.ml.tree._
import org.apache.spark.ml.tree.impl.GradientBoostedTrees
import org.apache.spark.ml.util._
import org.apache.spark.ml.util.DefaultParamsReader.Metadata
import org.apache.spark.ml.util.Instrumentation.instrumented
import org.apache.spark.mllib.tree.model.{GradientBoostedTreesModel => OldGBTModel}
import org.apache.spark.sql.{DataFrame, Dataset, Row}
import org.apache.spark.sql.functions._
import org.apache.spark.mllib.tree.configuration.{FeatureType, Algo => OldAlgo, BoostingStrategy => OldBoostingStrategy, Strategy => OldStrategy}
import org.apache.spark.sql.types.{DataType, DoubleType, StructType}

import scala.Array.range

/**
 * Gradient-Boosted Trees (GBTs) (http://en.wikipedia.org/wiki/Gradient_boosting)
 * learning algorithm for classification.
 * It supports binary labels, as well as both continuous and categorical features.
 *
 * The implementation is based upon: J.H. Friedman. "Stochastic Gradient Boosting." 1999.
 *
 * Notes on Gradient Boosting vs. TreeBoost:
 *  - This implementation is for Stochastic Gradient Boosting, not for TreeBoost.
 *  - Both algorithms learn tree ensembles by minimizing loss functions.
 *  - TreeBoost (Friedman, 1999) additionally modifies the outputs at tree leaf nodes
 * based on the loss function, whereas the original gradient boosting method does not.
 *  - We expect to implement TreeBoost in the future:
 * [https://issues.apache.org/jira/browse/SPARK-4240]
 *
 * @note Multiclass labels are not currently supported.
 */
@Since("1.4.0")
class GBTLeafClassifier @Since("1.4.0")(
                                         @Since("1.4.0") override val uid: String)
  extends ProbabilisticClassifier[Vector, GBTLeafClassifier, GBTClassificationLeafModel]
    with GbdtLeafParams with DefaultParamsWritable with Logging {

  @Since("1.4.0")
  def this() = this(Identifiable.randomUID("gbtc_leaf"))

  // Override parameter setters from parent trait for Java API compatibility.

  // Parameters from TreeClassifierParams:

  /** @group setParam */
  @Since("1.4.0")
  override def setMaxDepth(value: Int): this.type = set(maxDepth, value)

  /** @group setParam */
  @Since("1.4.0")
  override def setMaxBins(value: Int): this.type = set(maxBins, value)

  /** @group setParam */
  @Since("1.4.0")
  override def setMinInstancesPerNode(value: Int): this.type = set(minInstancesPerNode, value)

  /** @group setParam */
  @Since("1.4.0")
  override def setMinInfoGain(value: Double): this.type = set(minInfoGain, value)

  /** @group expertSetParam */
  @Since("1.4.0")
  override def setMaxMemoryInMB(value: Int): this.type = set(maxMemoryInMB, value)

  /** @group expertSetParam */
  @Since("1.4.0")
  override def setCacheNodeIds(value: Boolean): this.type = set(cacheNodeIds, value)

  /**
   * Specifies how often to checkpoint the cached node IDs.
   * E.g. 10 means that the cache will get checkpointed every 10 iterations.
   * This is only used if cacheNodeIds is true and if the checkpoint directory is set in
   * [[org.apache.spark.SparkContext]].
   * Must be at least 1.
   * (default = 10)
   *
   * @group setParam
   */
  @Since("1.4.0")
  override def setCheckpointInterval(value: Int): this.type = set(checkpointInterval, value)

  /**
   * The impurity setting is ignored for GBT models.
   * Individual trees are built using impurity "Variance."
   *
   * @group setParam
   */
  @Since("1.4.0")
  override def setImpurity(value: String): this.type = {
    logWarning("GBTClassifier.setImpurity should NOT be used")
    this
  }

  // Parameters from TreeEnsembleParams:

  /** @group setParam */
  @Since("1.4.0")
  override def setSubsamplingRate(value: Double): this.type = set(subsamplingRate, value)

  /** @group setParam */
  @Since("1.4.0")
  override def setSeed(value: Long): this.type = set(seed, value)

  // Parameters from GBTParams:

  /** @group setParam */
  @Since("1.4.0")
  override def setMaxIter(value: Int): this.type = set(maxIter, value)

  /** @group setParam */
  @Since("1.4.0")
  override def setStepSize(value: Double): this.type = set(stepSize, value)

  /** @group setParam */
  @Since("2.3.0")
  override def setFeatureSubsetStrategy(value: String): this.type =
    set(featureSubsetStrategy, value)

  // Parameters from GBTClassifierParams:

  /** @group setParam */
  @Since("1.4.0")
  def setLossType(value: String): this.type = set(lossType, value)

  /** @group setParam */
  @Since("2.4.0")
  def setValidationIndicatorCol(value: String): this.type = {
    set(validationIndicatorCol, value)
  }

  override def transformSchema(schema: StructType): StructType = {
    validateAndTransformSchema(schema, fitting = true, featuresDataType)
  }

  override protected def train(
                                dataset: Dataset[_]): GBTClassificationLeafModel = instrumented { instr =>
    val categoricalFeatures: Map[Int, Int] =
      MetadataUtils.getCategoricalFeatures(dataset.schema($(featuresCol)))

    val withValidation = isDefined(validationIndicatorCol) && $(validationIndicatorCol).nonEmpty

    // We copy and modify this from Classifier.extractLabeledPoints since GBT only supports
    // 2 classes now.  This lets us provide a more precise error message.
    val convert2LabeledPoint = (dataset: Dataset[_]) => {
      dataset.select(col($(labelCol)), col($(featuresCol))).rdd.map {
        case Row(label: Double, features: Vector) =>
          require(label == 0 || label == 1, s"GBTClassifier was given" +
            s" dataset with invalid label $label.  Labels must be in {0,1}; note that" +
            s" GBTClassifier currently only supports binary classification.")
          LabeledPoint(label, features)
      }
    }

    val (trainDataset, validationDataset) = if (withValidation) {
      (
        convert2LabeledPoint(dataset.filter(not(col($(validationIndicatorCol))))),
        convert2LabeledPoint(dataset.filter(col($(validationIndicatorCol))))
      )
    } else {
      (convert2LabeledPoint(dataset), null)
    }

    val numFeatures = trainDataset.first().features.size
    val boostingStrategy = super.getOldBoostingStrategy(categoricalFeatures, OldAlgo.Classification)

    val numClasses = 2
    if (isDefined(thresholds)) {
      require($(thresholds).length == numClasses, this.getClass.getSimpleName +
        ".train() called with non-matching numClasses and thresholds.length." +
        s" numClasses=$numClasses, but thresholds has length ${$(thresholds).length}")
    }

    instr.logPipelineStage(this)
    instr.logDataset(dataset)
    instr.logParams(this, labelCol, featuresCol, predictionCol, impurity, lossType,
      maxDepth, maxBins, maxIter, maxMemoryInMB, minInfoGain, minInstancesPerNode,
      seed, stepSize, subsamplingRate, cacheNodeIds, checkpointInterval, featureSubsetStrategy,
      validationIndicatorCol)
    instr.logNumFeatures(numFeatures)
    instr.logNumClasses(numClasses)

    val (baseLearners, learnerWeights) = if (withValidation) {
      GradientBoostedTrees.runWithValidation(trainDataset, validationDataset, boostingStrategy,
        $(seed), $(featureSubsetStrategy))
    } else {
      GradientBoostedTrees.run(trainDataset, boostingStrategy, $(seed), $(featureSubsetStrategy))
    }

    new GBTClassificationLeafModel(uid, baseLearners, learnerWeights, numFeatures)
  }

  @Since("1.4.1")
  override def copy(extra: ParamMap): GBTLeafClassifier = defaultCopy(extra)
}

@Since("1.4.0")
object GBTLeafClassifier extends DefaultParamsReadable[GBTLeafClassifier] {

  /** Accessor for supported loss settings: logistic */
  @Since("1.4.0")
  final val supportedLossTypes: Array[String] = GBTClassifierParams.supportedLossTypes

  @Since("2.0.0")
  override def load(path: String): GBTLeafClassifier = super.load(path)
}


/*
添加额外参数
 */
private[ml] trait GbdtLeafParams extends GBTClassifierParams {
  final val inputCol: Param[String] = new Param[String](this, "input", "输入column")

  @Since("2.4.0")
  def setInputCol(value: String): this.type = {
    set(inputCol, value)
  }

  final val outputCol: Param[String] = new Param[String](this, "output", "输出column")

  @Since("2.4.0")
  def setOutputCol(value: String): this.type = {
    set(outputCol, value)
  }


  override protected def validateAndTransformSchema(
                                                     schema: StructType,
                                                     fitting: Boolean,
                                                     featuresDataType: DataType): StructType = {
    val parentSchema = super.validateAndTransformSchema(schema, fitting, featuresDataType)
    SchemaUtils.appendColumn(parentSchema, $(outputCol), new VectorUDT)
  }
}

/**
 * Gradient-Boosted Trees (GBTs) (http://en.wikipedia.org/wiki/Gradient_boosting)
 * model for classification.
 * It supports binary labels, as well as both continuous and categorical features.
 *
 * @param _trees       Decision trees in the ensemble.
 * @param _treeWeights Weights for the decision trees in the ensemble.
 * @note Multiclass labels are not currently supported.
 */
@Since("1.6.0")
class GBTClassificationLeafModel private[ml](
                                              @Since("1.6.0") override val uid: String,
                                              private val _trees: Array[DecisionTreeRegressionModel],
                                              private val _treeWeights: Array[Double],
                                              @Since("1.6.0") override val numFeatures: Int,
                                              @Since("2.2.0") override val numClasses: Int)
  extends ProbabilisticClassificationModel[Vector, GBTClassificationLeafModel]
    with GbdtLeafParams with TreeEnsembleModel[DecisionTreeRegressionModel]
    with MLWritable with Serializable {

  require(_trees.nonEmpty, "GBTClassificationModel requires at least 1 tree.")
  require(_trees.length == _treeWeights.length, "GBTClassificationModel given trees, treeWeights" +
    s" of non-matching lengths (${_trees.length}, ${_treeWeights.length}, respectively).")

  /**
   * Construct a GBTClassificationModel
   *
   * @param _trees       Decision trees in the ensemble.
   * @param _treeWeights Weights for the decision trees in the ensemble.
   * @param numFeatures  The number of features.
   */
  private[ml] def this(
                        uid: String,
                        _trees: Array[DecisionTreeRegressionModel],
                        _treeWeights: Array[Double],
                        numFeatures: Int) =
    this(uid, _trees, _treeWeights, numFeatures, 2)

  /**
   * Construct a GBTClassificationModel
   *
   * @param _trees       Decision trees in the ensemble.
   * @param _treeWeights Weights for the decision trees in the ensemble.
   */
  @Since("1.6.0")
  def this(uid: String, _trees: Array[DecisionTreeRegressionModel], _treeWeights: Array[Double]) =
    this(uid, _trees, _treeWeights, -1, 2)

  @Since("1.4.0")
  override def trees: Array[DecisionTreeRegressionModel] = _trees

  /**
   * Number of trees in ensemble
   */
  @Since("2.0.0")
  val getNumTrees: Int = trees.length

  @Since("1.4.0")
  override def treeWeights: Array[Double] = _treeWeights

  // todo 改写这个 transform方法，输出结果为叶子节点向量值

  import GBTLRUtil._

  val treeLeafArray: Array[Array[Double]] = getTreeLeafArray(trees)

  override protected def transformImpl(dataset: Dataset[_]): DataFrame = {
    val bcastModel = dataset.sparkSession.sparkContext.broadcast(this)
    val predictUDF = udf { (features: Any) =>
      bcastModel.value.predict(features.asInstanceOf[Vector])
    }

    val predictRawUDF = udf { (features: Any) =>
      bcastModel.value.predictRaw(features.asInstanceOf[Vector])
    }

    val probabilityUDF = udf { (rawP: Any) =>
      bcastModel.value.raw2probabilityInPlace(rawP.asInstanceOf[Vector])
    }

    val getLeafUDF = udf { (features: Any) =>
      var gbtFeatures = new Array[Double](0)
      for (i <- range(0, numTrees)) {
        val loc = gbtPredict(bcastModel.value.trees(i).toOld.topNode, features.asInstanceOf[Vector].toDense) //样本点所落叶节点位置
        val leafArray = new Array[Double](bcastModel.value.trees(i).numNodes / 2 + 1) //满员二叉树叶节点数=(总结点数/2)+1
        leafArray(treeLeafArray(i).indexOf(loc)) = 1 //val数组元素可变，长度不可变
        gbtFeatures = gbtFeatures ++ leafArray //++两边要求数据类型一致
      }

      //      println("==========transformImpl===========")
      //      println(s"gbtFeatures: ${gbtFeatures.deep.mkString("\n")}")
      Vectors.dense(gbtFeatures).toSparse
    }


    //    dataset.withColumn($(predictionCol), getLeafUDF(col($(featuresCol))))
    dataset.withColumn($ {
      outputCol
    }, getLeafUDF(col($(featuresCol))))
      .withColumn($(predictionCol), predictUDF(col($(featuresCol))))
      .withColumn($(rawPredictionCol), predictRawUDF(col($(featuresCol))))
      .withColumn($(probabilityCol), probabilityUDF(col($(rawPredictionCol))))
  }

  @org.apache.spark.annotation.Since("2.0.0")
  override def transform(dataset: Dataset[_]): DataFrame = {
    transformImpl(dataset)
  }


  override def predict(features: Vector): Double = {
    // If thresholds defined, use predictRaw to get probabilities, otherwise use optimization
    if (isDefined(thresholds)) {
      super.predict(features)
    } else {
      if (margin(features) > 0.0) 1.0 else 0.0
    }
  }

  override protected def predictRaw(features: Vector): Vector = {
    val prediction: Double = margin(features)
    Vectors.dense(Array(-prediction, prediction))
  }

  override protected def raw2probabilityInPlace(rawPrediction: Vector): Vector = {
    rawPrediction match {
      case dv: DenseVector =>
        dv.values(0) = loss.computeProbability(dv.values(0))
        dv.values(1) = 1.0 - dv.values(0)
        dv
      case sv: SparseVector =>
        throw new RuntimeException("Unexpected error in GBTClassificationModel:" +
          " raw2probabilityInPlace encountered SparseVector")
    }
  }

  /** Number of trees in ensemble */
  val numTrees: Int = trees.length

  @Since("1.4.0")
  override def copy(extra: ParamMap): GBTClassificationLeafModel = {
    copyValues(new GBTClassificationLeafModel(uid, _trees, _treeWeights, numFeatures, numClasses),
      extra).setParent(parent)
  }

  @Since("1.4.0")
  override def toString: String = {
    s"GBTClassificationModel (uid=$uid) with $numTrees trees"
  }

  /**
   * Estimate of the importance of each feature.
   *
   * Each feature's importance is the average of its importance across all trees in the ensemble
   * The importance vector is normalized to sum to 1. This method is suggested by Hastie et al.
   * (Hastie, Tibshirani, Friedman. "The Elements of Statistical Learning, 2nd Edition." 2001.)
   * and follows the implementation from scikit-learn.
   *
   * See `DecisionTreeClassificationModel.featureImportances`
   */
  @Since("2.0.0")
  lazy val featureImportances: Vector = TreeEnsembleModel.featureImportances(trees, numFeatures)

  /** Raw prediction for the positive class. */
  private def margin(features: Vector): Double = {
    val treePredictions = _trees.map(_.rootNode.predictImpl(features).prediction)
    blas.ddot(numTrees, treePredictions, 1, _treeWeights, 1)
  }

  /** (private[ml]) Convert to a model in the old API */
  private[ml] def toOld: OldGBTModel = {
    new OldGBTModel(OldAlgo.Classification, _trees.map(_.toOld), _treeWeights)
  }

  // hard coded loss, which is not meant to be changed in the model
  private val loss = getOldLossType

  /**
   * Method to compute error or loss for every iteration of gradient boosting.
   *
   * @param dataset Dataset for validation.
   */
  @Since("2.4.0")
  def evaluateEachIteration(dataset: Dataset[_]): Array[Double] = {
    val data = dataset.select(col($(labelCol)), col($(featuresCol))).rdd.map {
      case Row(label: Double, features: Vector) => LabeledPoint(label, features)
    }
    GradientBoostedTrees.evaluateEachIteration(data, trees, treeWeights, loss,
      OldAlgo.Classification
    )
  }

  @Since("2.0.0")
  override def write: MLWriter = new GBTClassificationLeafModel.GBTClassificationModelWriter(this)
}

@Since("2.0.0")
object GBTClassificationLeafModel extends MLReadable[GBTClassificationLeafModel] {

  private val numFeaturesKey: String = "numFeatures"
  private val numTreesKey: String = "numTrees"

  @Since("2.0.0")
  override def read: MLReader[GBTClassificationLeafModel] = new GBTClassificationModelReader

  @Since("2.0.0")
  override def load(path: String): GBTClassificationLeafModel = super.load(path)

  private[GBTClassificationLeafModel]
  class GBTClassificationModelWriter(instance: GBTClassificationLeafModel) extends MLWriter {

    override protected def saveImpl(path: String): Unit = {

      val extraMetadata: JObject = Map(
        numFeaturesKey -> instance.numFeatures,
        numTreesKey -> instance.getNumTrees)
      EnsembleModelReadWrite.saveImpl(instance, path, sparkSession, extraMetadata)
    }
  }

  private class GBTClassificationModelReader extends MLReader[GBTClassificationLeafModel] {

    /** Checked against metadata when loading model */
    private val className = classOf[GBTClassificationLeafModel].getName
    private val treeClassName = classOf[DecisionTreeRegressionModel].getName

    override def load(path: String): GBTClassificationLeafModel = {
      implicit val format = DefaultFormats
      val (metadata: Metadata, treesData: Array[(Metadata, Node)], treeWeights: Array[Double]) =
        EnsembleModelReadWrite.loadImpl(path, sparkSession, className, treeClassName)
      val numFeatures = (metadata.metadata \ numFeaturesKey).extract[Int]
      val numTrees = (metadata.metadata \ numTreesKey).extract[Int]

      val trees: Array[DecisionTreeRegressionModel] = treesData.map {
        case (treeMetadata, root) =>
          val tree =
            new DecisionTreeRegressionModel(treeMetadata.uid, root, numFeatures)
          treeMetadata.getAndSetParams(tree)
          tree
      }
      require(numTrees == trees.length, s"GBTClassificationModel.load expected $numTrees" +
        s" trees based on metadata but found ${trees.length} trees.")
      val model = new GBTClassificationLeafModel(metadata.uid,
        trees, treeWeights, numFeatures)
      metadata.getAndSetParams(model)
      model
    }
  }

  /** Convert a model from the old API */
  private[ml] def fromOld(
                           oldModel: OldGBTModel,
                           parent: GBTLeafClassifier,
                           categoricalFeatures: Map[Int, Int],
                           numFeatures: Int = -1,
                           numClasses: Int = 2): GBTClassificationLeafModel = {
    require(oldModel.algo == OldAlgo.Classification, "Cannot convert GradientBoostedTreesModel" +
      s" with algo=${oldModel.algo} (old API) to GBTClassificationModel (new API).")
    val newTrees = oldModel.trees.map { tree =>
      // parent for each tree is null since there is no good way to set this.
      DecisionTreeRegressionModel.fromOld(tree, null, categoricalFeatures)
    }
    val uid = if (parent != null) parent.uid else Identifiable.randomUID("gbtc")
    new GBTClassificationLeafModel(uid, newTrees, oldModel.treeWeights, numFeatures, numClasses)
  }
}


object GBTLRUtil {
  //获取一棵树的所有叶节点
  def getLeafNode(node: MlibNode): Array[Double] = {
    var leafNode = Array[Double]() //
    if (node.isLeaf) {
      leafNode = leafNode :+ node.id.toDouble
    }
    else {
      leafNode = leafNode ++ getLeafNode(node.leftNode.get) //数组与数组相加，元素相加，内部均由数组元素组成
      leafNode = leafNode ++ getLeafNode(node.rightNode.get)
    }
    leafNode //返回值
  }

  //返回样本所在叶节点
  def gbtPredict(node: MlibNode, features: DenseVector): Double = {
    //树模型在非叶节点的分割点
    val split = node.split

    if (node.isLeaf) {
      node.id
    }
    else {
      //判断连续或者离散，不同的特征判断方式
      if (split.get.featureType == FeatureType.Continuous) {
        //split.get.feature数据类型为int，表示feature index
        if (features(split.get.feature) <= split.get.threshold) {
          gbtPredict(node.leftNode.get, features)
        }
        else {
          gbtPredict(node.rightNode.get, features)
        }
      }
      else {
        //categories默认为左子节点特征列表
        if (split.get.categories.contains(features(split.get.feature))) {
          gbtPredict(node.leftNode.get, features)
        }
        else {
          gbtPredict(node.rightNode.get, features)
        }
      }
    }
  }

  def getTreeLeafArray(trees: Array[DecisionTreeRegressionModel]): Array[Array[Double]] = {
    val numTrees = trees.length
    val treeLeafArray = new Array[Array[Double]](numTrees)
    //    trees.foreach(f => {
    //      println(f.rootNode)
    //      //    println(f.rootNode.impurity)
    //    })
    //    println(s" tree length:${numTrees}    treeLeafArray:${treeLeafArray.deep.mkString("\n")}")
    //存储叶节点的index
    for (i <- range(0, trees.length)) {
      //      println(i)
      //      println(trees(i).rootNode)
      treeLeafArray(i) = getLeafNode(trees(i).toOld.topNode)
    }

    //    println(s" tree length:${numTrees}    treeLeafArray:${treeLeafArray.deep.mkString("\n")}")
    treeLeafArray
  }
}

