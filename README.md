# gbdt_leaf_spark(gbdt+lr)

    
    object GbdtLeafTest {
      val modelPath = ""
      val mleapModelPath = ""
    
      def loadModel(spark: SparkSession, df: DataFrame): PipelineModel = {
    
        val sameModel = PipelineModel.load(modelPath)
    
        sameModel.transform(df).show(false)
        sameModel
      }
    
      def saveModel(spark: SparkSession, df: DataFrame): Unit = {
        var model_stages = new collection.mutable.ListBuffer[PipelineStage]()
    
        val transformer = new StringIndexer()
          .setInputCol("ad__cr_id")
          .setOutputCol("ad__cr_id_1")
          .setHandleInvalid("keep")
    
        model_stages += transformer
    
        val transformer2 = new StringIndexer()
          .setInputCol("gender")
          .setOutputCol("gender_1")
          .setHandleInvalid("keep")
    
        model_stages += transformer2
    
        val inter =
          new VectorAssembler().setInputCols(Array("ad__cr_id_1", "gender_1")).setOutputCol("gbdt_features")
        model_stages += inter
        val samplePipline = new Pipeline()
          .setStages(model_stages.toArray)
          .fit(df)
        val trainData = samplePipline.transform(df)
    
        val gbt = new GBTLeafClassifier()
          .setLabelCol("label")
          .setFeaturesCol("gbdt_features")
          .setOutputCol("gbdtGenFeatures1")
          .setMaxIter(1)
          .setPredictionCol("gbdt_prediction")
          .setRawPredictionCol("gbdt_predictiton_raw")
          .setProbabilityCol("gbdt_prob")
          .setMaxBins(1000)
          .setMaxDepth(3)
          .setMinInfoGain(0.01)
        model_stages += gbt
        //    gbt.fit(trainDataForGbdt)
    
        val onlinePipeline =
          new Pipeline()
            .setStages(model_stages.toArray)
            .fit(df)
        val trainDataForGbdt = onlinePipeline.transform(df)
        trainDataForGbdt.show(false)
    
        //    val f = gbt.fit(trainData)
        //    onlinePipeline.transform(trainData).show(false)
        //    val g = new GBTClassificationLeafModel(new GBTClassificationLeafModel)
        onlinePipeline.write.overwrite().save(modelPath)
        //    f.write.overwrite().save(modelPath)
      }
    
      def getTrainData(spark: SparkSession, df: DataFrame): DataFrame = {
        var model_stages = new collection.mutable.ListBuffer[PipelineStage]()
    
        val transformer = new StringIndexer()
          .setInputCol("ad__cr_id")
          .setOutputCol("ad__cr_id_1")
          .setHandleInvalid("keep")
    
        model_stages += transformer
    
        val transformer2 = new StringIndexer()
          .setInputCol("gender")
          .setOutputCol("gender_1")
          .setHandleInvalid("keep")
    
        model_stages += transformer2
    
        val inter =
          new Interaction().setInputCols(Array("ad__cr_id_1", "gender_1")).setOutputCol("features")
        model_stages += inter
        val samplePipline = new Pipeline()
          .setStages(model_stages.toArray)
          .fit(df)
        val trainData = samplePipline.transform(df)
        trainData
      }
    
      def testGbdtLR(spark: SparkSession, df: DataFrame): Unit = {
        var model_stages = new collection.mutable.ListBuffer[PipelineStage]()
    
    
        // 特征工程
        // 1.load gbdt
        val gbdtPipline = loadModel(spark, df)
    
        gbdtPipline.transform(df).show(false)
        println(s"gbdt pipline ${gbdtPipline.stages}")
        model_stages ++= gbdtPipline.stages
        println(s"after add gbdt length ${model_stages.length}")
        new Pipeline()
          .setStages(model_stages.toArray)
          .fit(df).transform(df).show(false)
    
        val transformer3 = new VectorAssembler().setInputCols(Array("ad__cr_id_1", "gender_1", "gbdtGenFeatures1")).setOutputCol("features")
        model_stages += transformer3
    
    
        val lr = new LogisticRegression()
          .setMaxIter(2)
          .setTol(0.1)
          .setElasticNetParam(0.01)
          .setFeaturesCol("gbdtGenFeatures1")
          .setLabelCol("label")
    
        model_stages += lr
        val onlinePipline = new Pipeline()
          .setStages(model_stages.toArray)
          .fit(df)
    
        onlinePipline.transform(df).show(false)
      }
    
      def main(args: Array[String]): Unit = {
        val conf = new SparkConf().setMaster("local").setAppName("Test Application")
        val spark =
          SparkSession.builder().config(conf).getOrCreate()
        val df = spark.read
          .option("header", "true")
          .option("inferSchema", "true")
          .option("delimiter", "\t")
          .csv("src/main/scala/org.apache.spark.ml.mleap.gbdt/data.csv") //.select("ad__cr_id", "label", "gender")
        spark.sparkContext.setLogLevel("WARN")
        saveModel(spark, df)
        loadModel(spark, df)
        testGbdtLR(spark, df)
      }
    
    }
