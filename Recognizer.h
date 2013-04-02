#include "features.h"
#include <pcl/filters/extract_indices.h>

//#define DEBUG
//#define SEGSHOW
#define THREADNUM 8

class SegRecog{
public:
	SegRecog(std::string model_path, float model_sample_rate_);
	//SegRecog(pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr dense_model_, pcl::PointCloud<PointObj>::Ptr model_sift_ori_, pcl::PointCloud<SIFTDescr>::Ptr model_SIFTDescr_, float resolution_);
	~SegRecog(){};

	void LoadScene(pcl::PointCloud<PointXYZRGBIM>::Ptr scene_, pcl::PointCloud<NormalT>::Ptr scene_normals_);
	void LoadScene(const std::vector<pcl::PointCloud<PointT>::Ptr, Eigen::aligned_allocator<pcl::PointCloud<PointT>::Ptr>> &segs_rgb_,
					const std::vector<pcl::PointCloud<NormalT>::Ptr, Eigen::aligned_allocator<pcl::PointCloud<NormalT>::Ptr>> &segs_normals_,
					const std::vector<pcl::PointCloud<PointT>::Ptr, Eigen::aligned_allocator<pcl::PointCloud<PointT>::Ptr>> &segs_shot_rgb_,
					const std::vector<pcl::PointCloud<NormalT>::Ptr, Eigen::aligned_allocator<pcl::PointCloud<NormalT>::Ptr>> &segs_shot_normals_,
					const std::vector<pcl::PointCloud<SHOTdescr>::Ptr, Eigen::aligned_allocator<pcl::PointCloud<SHOTdescr>::Ptr>> &segs_SHOTDescr_,
					const std::vector<pcl::PointCloud<PointT>::Ptr, Eigen::aligned_allocator<pcl::PointCloud<PointT>::Ptr>> &segs_sift_rgb_,
					const std::vector<pcl::PointCloud<NormalT>::Ptr, Eigen::aligned_allocator<pcl::PointCloud<NormalT>::Ptr>> &segs_sift_normals_,
					const std::vector<pcl::PointCloud<SIFTDescr>::Ptr, Eigen::aligned_allocator<pcl::PointCloud<SIFTDescr>::Ptr>> &segs_SIFTDescr_);

	// Shared by different recognizers
	std::vector<pcl::PointCloud<PointT>::Ptr, Eigen::aligned_allocator<pcl::PointCloud<PointT>::Ptr>> segs_rgb;
	std::vector<pcl::PointCloud<NormalT>::Ptr, Eigen::aligned_allocator<pcl::PointCloud<NormalT>::Ptr>> segs_normals;
	std::vector<pcl::PointCloud<PointT>::Ptr, Eigen::aligned_allocator<pcl::PointCloud<PointT>::Ptr>> segs_shot_rgb;
	std::vector<pcl::PointCloud<NormalT>::Ptr, Eigen::aligned_allocator<pcl::PointCloud<NormalT>::Ptr>> segs_shot_normals;
	std::vector<pcl::PointCloud<SHOTdescr>::Ptr, Eigen::aligned_allocator<pcl::PointCloud<SHOTdescr>::Ptr>> segs_SHOTDescr;
	std::vector<pcl::PointCloud<PointT>::Ptr, Eigen::aligned_allocator<pcl::PointCloud<PointT>::Ptr>> segs_sift_rgb;
	std::vector<pcl::PointCloud<NormalT>::Ptr, Eigen::aligned_allocator<pcl::PointCloud<NormalT>::Ptr>> segs_sift_normals;
	std::vector<pcl::PointCloud<SIFTDescr>::Ptr, Eigen::aligned_allocator<pcl::PointCloud<SIFTDescr>::Ptr>> segs_SIFTDescr;

	void setModelSR(float model_sample_ratio_){model_sample_ratio = model_sample_ratio_;}
	void setPartSR(float part_sample_ratio_){part_sample_ratio = part_sample_ratio_;}
	void setDistT(float distT_){distT = distT_;}
	void setCurveT(float curveT_){curveT = curveT_;}
	void setHueT(float hueT_){hueT = hueT_;}
	void setCorrsRatio(float corrs_ratio_){corrs_ratio = corrs_ratio_;}
	void setConEpsilon(float converge_epsilon_){converge_epsilon = converge_epsilon_;}
	void setHistT(float histT_){histT = histT_;}
	void setSIFTRatio(float sift_ratio_){sift_ratio = sift_ratio_;}

	void DisableSIFT(){sift_engine = false;}
	void DisableSHOT(){shot_engine = false;}
	
	void setMinSegment(float minSegment_){minSegment = minSegment_;}
	void setMaxSegment(float maxSegment_){maxSegment = maxSegment_;}
	void setSearchNeighs(float searchNeighs_){searchNeighs = searchNeighs_;}
	void setNormalDiff(float normalDiff_){normalDiff = normalDiff_;} 

	// Recognition funcion
	void Recognize();

	std::vector<Eigen::Matrix4f, Eigen::aligned_allocator<Eigen::Matrix4f>> GetReocgResult(){return recog_result;}
	pcl::PointCloud<PointT>::Ptr GetModelMesh(){return model_rgb;}

private:
	float model_sample_ratio;
	float part_sample_ratio;
	float SIFT_iter_ratio;

	// For dense matching
	float distT;			//resolution * 7
	float curveT;			//0.03
	float hueT;				//0.08
	float corrs_ratio;		//0.5
	float converge_epsilon;	//0.000001

	// For region HSI histogram
	float histT;			//0.3
	int h_bin;				//32
	int s_bin;				//16
	int i_bin;				//16
	
	float resolution;
	float sift_ratio;		//0.7

	//For Segmentation
	int minSegment;			//500
	int maxSegment;			//100000
	int searchNeighs;		//10
	float normalDiff;		//1.8

	bool sift_engine;
	bool shot_engine;

#if defined(DEBUG) || defined(SEGSHOW)
	pcl::visualization::PCLVisualizer::Ptr viewer;
#endif

	std::vector<Eigen::Matrix4f, Eigen::aligned_allocator<Eigen::Matrix4f>> recog_result;

	// Model
	pcl::PointCloud<PointT>::Ptr model_rgb;
	pcl::PointCloud<NormalT>::Ptr model_normals;
	pcl::PointCloud<PointT>::Ptr model_shot;
	pcl::PointCloud<NormalT>::Ptr model_shot_normals;
	pcl::PointCloud<SHOTdescr>::Ptr model_SHOTDescr;
	pcl::PointCloud<PointT>::Ptr model_sift;
	pcl::PointCloud<NormalT>::Ptr model_sift_normals;
	pcl::PointCloud<SIFTDescr>::Ptr model_SIFTDescr;

	//Utilities
	float distPt(const PointT &pt1, const PointT &pt2);
	bool testTransPt(pcl::PointCloud<PointT>::Ptr src, pcl::PointCloud<PointT>::Ptr dst, float inlierT);
	void GenNormalPt(const pcl::PointCloud<PointT>::Ptr cloud, const pcl::PointCloud<NormalT>::Ptr cloud_normals, std::vector<int> idx, pcl::PointCloud<PointT>::Ptr candidates);

	// Dense matching functions
	pcl::CorrespondencesPtr testGuess(pcl::PointCloud<pcl::PointXYZHSV>::Ptr part_hue_, pcl::PointCloud<pcl::PointXYZHSV>::Ptr model_hue_, pcl::PointCloud<NormalT>::Ptr part_normals_, pcl::PointCloud<NormalT>::Ptr model_normals_, float hueT, float curveT, float distT);
	float myICP(pcl::PointCloud<PointT>::Ptr model_rgb, pcl::PointCloud<PointT>::Ptr scene_rgb, pcl::PointCloud<pcl::PointXYZHSV>::Ptr model_hue, pcl::PointCloud<pcl::PointXYZHSV>::Ptr scene_hue, pcl::PointCloud<NormalT>::Ptr model_normals, pcl::PointCloud<NormalT>::Ptr scene_normals, 
		const Eigen::Matrix4f &initial_guess, Eigen::Matrix4f &rotatranslation);
	
	// Fast hypothesis verification
	float testShapeDist(const pcl::PointCloud<PointT>::Ptr cloud, const pcl::search::KdTree<PointT>::Ptr tree);

	// SHOT recognition pipeline
	Eigen::Matrix4f recogSHOT(pcl::PointCloud<PointT>::Ptr part, pcl::PointCloud<PointT>::Ptr model, pcl::PointCloud<NormalT>::Ptr part_normals, pcl::PointCloud<NormalT>::Ptr model_normals,
			pcl::PointCloud<SHOTdescr>::Ptr part_shot, pcl::PointCloud<SHOTdescr>::Ptr model_shot, pcl::PointCloud<PointT>::Ptr dense_model, pcl::PointCloud<NormalT>::Ptr dense_normals);
	std::vector<std::vector<int>> List_SHOT_Corrs(pcl::PointCloud<SHOTdescr>::Ptr part_shot, pcl::PointCloud<SHOTdescr>::Ptr model_shot);
	Eigen::Matrix4f poseSHOTGenerator(pcl::PointCloud<PointT>::Ptr part_, pcl::PointCloud<PointT>::Ptr model_, pcl::PointCloud<NormalT>::Ptr part_normals_, pcl::PointCloud<NormalT>::Ptr model_normals_, 
				pcl::PointCloud<PointT>::Ptr dense_model_, const std::vector<std::vector<int>> &corrsPool_);

	//SIFT recognition piepline
	Eigen::Matrix4f recogSIFT(pcl::PointCloud<PointT>::Ptr part_sift, pcl::PointCloud<NormalT>::Ptr part_sift_normals, pcl::PointCloud<SIFTDescr>::Ptr part_SIFTDescr, 
						pcl::PointCloud<PointT>::Ptr model_sift, pcl::PointCloud<NormalT>::Ptr model_sift_normals, pcl::PointCloud<SIFTDescr>::Ptr model_SIFTDescr, 
						pcl::PointCloud<PointT>::Ptr part, pcl::PointCloud<NormalT>::Ptr part_normals, pcl::PointCloud<PointT>::Ptr dense_model, pcl::PointCloud<NormalT>::Ptr dense_normals);
	std::vector<std::vector<int>> List_SIFT_Corrs(pcl::PointCloud<SIFTDescr>::Ptr part_SIFTDescr, pcl::PointCloud<SIFTDescr>::Ptr model_SIFTDescr);
	Eigen::Matrix4f poseSIFTGenerator(pcl::PointCloud<PointT>::Ptr in_part_sift, pcl::PointCloud<PointT>::Ptr model_sift, pcl::PointCloud<NormalT>::Ptr in_part_sift_normals, pcl::PointCloud<NormalT>::Ptr model_sift_normals, 
				pcl::PointCloud<PointT>::Ptr part, pcl::PointCloud<PointT>::Ptr dense_model, const std::vector<std::vector<int>> &corrsPool);

	//Point Cloud Segmentation
	void extractSegments(pcl::PointCloud<PointT>::Ptr scene_rgb, pcl::PointCloud<NormalT>::Ptr scene_normals, 
		pcl::PointCloud<PointT>::Ptr scene_sift, pcl::PointCloud<NormalT>::Ptr scene_sift_normals, pcl::PointCloud<SIFTDescr>::Ptr scene_SIFTDescr, std::vector<int> sift_idx,
		std::vector<pcl::PointCloud<PointT>::Ptr, Eigen::aligned_allocator<pcl::PointCloud<PointT>::Ptr>> &segs_rgb,
		std::vector<pcl::PointCloud<NormalT>::Ptr, Eigen::aligned_allocator<pcl::PointCloud<NormalT>::Ptr>> &segs_normals,
		std::vector<pcl::PointCloud<PointT>::Ptr, Eigen::aligned_allocator<pcl::PointCloud<PointT>::Ptr>> &segs_shot_rgb,
		std::vector<pcl::PointCloud<NormalT>::Ptr, Eigen::aligned_allocator<pcl::PointCloud<NormalT>::Ptr>> &segs_shot_normals,
		std::vector<pcl::PointCloud<SHOTdescr>::Ptr, Eigen::aligned_allocator<pcl::PointCloud<SHOTdescr>::Ptr>> &segs_SHOTDescr,
		std::vector<pcl::PointCloud<PointT>::Ptr, Eigen::aligned_allocator<pcl::PointCloud<PointT>::Ptr>> &segs_sift_rgb,
		std::vector<pcl::PointCloud<NormalT>::Ptr, Eigen::aligned_allocator<pcl::PointCloud<NormalT>::Ptr>> &segs_sift_normals,
		std::vector<pcl::PointCloud<SIFTDescr>::Ptr, Eigen::aligned_allocator<pcl::PointCloud<SIFTDescr>::Ptr>> &segs_SIFTDescr);
	
	std::vector <pcl::PointIndices> sceneSegmentation(pcl::PointCloud<PointT>::Ptr scene_rgb, pcl::PointCloud<NormalT>::Ptr scene_normals);
};

class VotingRecog{
public:
	VotingRecog(std::string modelname, bool refined_keypoints);
	~VotingRecog();

	void LoadScene(pcl::PointCloud<PointXYZRGBIM>::Ptr scene_, pcl::PointCloud<NormalT>::Ptr scene_normals_);

	void LoadScene(pcl::PointCloud<PointT>::Ptr scene_rgb_, pcl::PointCloud<NormalT>::Ptr scene_normals_, 
				pcl::PointCloud<PointT>::Ptr scene_keypoints_, pcl::PointCloud<NormalT>::Ptr scene_keypoints_normals_, 
				pcl::PointCloud<pcl::ReferenceFrame>::Ptr scene_rf_, pcl::PointCloud<SIFTDescr>::Ptr scene_SIFTDescr_, pcl::PointCloud<Descriptor3DType>::Ptr scene_CSHOTDescr_);

	void Recognize();

	void Reset();

	void setSIFTRatio(float sift_ratio_){	sift_ratio = sift_ratio_;	}
	void setNormalRatio(float normal_ratio_){	normal_ratio = normal_ratio_;	}
	void setResolution(float resolution_){	resolution = resolution_;	}
	void setBinSize(float bin_size_)	{	bin_size = bin_size_;	}
	void setRFRatio(float rf_ratio_){	rf_ratio = rf_ratio_;	}
	void setCurvature(float curvature_threshold_){	curvature_threshold = curvature_threshold_;	}
	void setInlierNumber(int inlier_threshold_){	inlier_threshold = inlier_threshold_;	}
	void setShowVoting(bool show_voting_){	show_voting = show_voting_;	}
	void setShowSurface(bool show_surface_){	show_surface = show_surface_;	}
	void setPCLViewer(pcl::visualization::PCLVisualizer::Ptr viewer_){	PCLViewer = true; viewer = viewer_;	}
	void setRefinedKeypoints(bool refined_keypoints_){	refined_keypoints = refined_keypoints_;	}
	void setSamplingRate(float sampling_rate_){ sampling_rate = sampling_rate_; }
	void setCSHOTRatio(float CSHOT_ratio_){ CSHOT_ratio = CSHOT_ratio_; }
	void setCSHOTThreshold(float CSHOT_threshold_){ CSHOT_threshold = CSHOT_threshold_; }

	pcl::PointCloud<PointT>::Ptr scene_rgb;
	pcl::PointCloud<NormalT>::Ptr scene_normals;
	pcl::PointCloud<PointT>::Ptr scene_keypoints;
	pcl::PointCloud<NormalT>::Ptr scene_keypoints_normals;
	pcl::PointCloud<pcl::ReferenceFrame>::Ptr scene_rf;
	pcl::PointCloud<SIFTDescr>::Ptr scene_SIFTDescr; 
	pcl::PointCloud<Descriptor3DType>::Ptr scene_CSHOTDescr; 
	//cv::Mat scene_descr;
	
	std::vector<pcl::CorrespondencesPtr, Eigen::aligned_allocator<pcl::CorrespondencesPtr>> corrs_clusters;
	pcl::PointCloud<PointT>::Ptr matched_model_keypoints;     
	pcl::PointCloud<PointT>::Ptr matched_scene_keypoints;
	
	std::vector<pcl::PointCloud<PointT>::Ptr, Eigen::aligned_allocator<pcl::PointCloud<PointT>::Ptr>> candidates;
	std::vector<Eigen::Matrix4f, Eigen::aligned_allocator<Eigen::Matrix4f>> transform_vec;
	std::vector<float> fitness_score_vec;

private:
	
	float sift_ratio;
	float normal_ratio;
	float resolution;
	float bin_size;
	float rf_ratio;
	float curvature_threshold;
	int inlier_threshold;
	float sampling_rate;
	float CSHOT_ratio;
	float CSHOT_threshold;

	bool PCLViewer;
	bool show_voting;
	bool show_surface;
	bool refined_keypoints;

	pcl::visualization::PCLVisualizer::Ptr viewer;

	//pcl::PointCloud<PointObj>::Ptr model;
	//pcl::PointCloud<PointT>::Ptr model_rgb;
	//pcl::PointCloud<NormalT>::Ptr model_normals;
	pcl::PointCloud<PointT>::Ptr reduced_model_rgb;
	pcl::PointCloud<NormalT>::Ptr reduced_model_normals;

	//pcl::PointCloud<PointObj>::Ptr model_keypoints_ori;
	pcl::PointCloud<PointT>::Ptr model_keypoints;
	pcl::PointCloud<NormalT>::Ptr model_keypoints_normals;
	pcl::PointCloud<pcl::PointXYZ>::Ptr model_shift;
	pcl::PointCloud<SIFTDescr>::Ptr model_SIFTDescr; 
	pcl::PointCloud<Descriptor3DType>::Ptr model_CSHOTDescr; 
	//cv::Mat model_descr;
	cv::Mat model_sift_num;

	pcl::PointCloud<pcl::PointXYZ>::Ptr votes;
	pcl::CorrespondencesPtr matched_corrs;

	void ComputeCloudRF(pcl::PointCloud<PointT>::Ptr keypoints, pcl::PointCloud<NormalT>::Ptr surface_normals, 
					pcl::PointCloud<PointT>::Ptr surface, pcl::PointCloud<pcl::ReferenceFrame>::Ptr keypoints_rf, float rf_rad);

	cv::Mat ExtractRowsCVMat(cv::Mat src, int s_idx, int e_idx);

	void ExtractSubDescr(pcl::PointCloud<SIFTDescr>::Ptr ori, pcl::PointCloud<SIFTDescr>::Ptr cur, int s_idx, int e_idx);

	std::vector<int> GetRawSIFT(pcl::PointCloud<PointXYZRGBIM>::Ptr model, pcl::PointCloud<NormalT>::Ptr model_normals, std::vector<cv::KeyPoint> &model_sift_keypoints, cv::Mat model_2DTo3D, cv::Mat pre_sift_descriptors, 
		pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr sift_cloud, cv::Mat &cur_sift_descriptors);
	
	bool ComputeSIFT(pcl::PointCloud<myPointT>::Ptr mycloud, cv::Mat& cloud_image, cv::Mat& cloud_2DTo3D, std::vector<cv::KeyPoint>& cloud_sift_keypoints, cv::Mat& cloud_sift_descriptors);

	void computeNormals(pcl::PointCloud<PointT>::Ptr cloud, pcl::PointCloud<NormalT>::Ptr cloud_normals, float normal_ss);

	void computeKeyNormals(pcl::PointCloud<PointT>::Ptr keypoints, pcl::PointCloud<NormalT>::Ptr keypoints_normals, pcl::PointCloud<PointT>::Ptr surface, float normal_ss);

	void FindSurface(pcl::PointCloud<PointT>::Ptr keypoints,pcl::PointCloud<PointT>::Ptr scene_rgb, pcl::PointCloud<NormalT>::Ptr scene_normals, pcl::PointCloud<PointT>::Ptr surface);

	void Match3DSIFT(pcl::PointCloud<SIFTDescr>::Ptr descr1, pcl::PointCloud<SIFTDescr>::Ptr descr2, pcl::CorrespondencesPtr scene_model_corrs, float *sim_score);
	
	void ExtractBestCorrs(pcl::CorrespondencesPtr scene_model_corrs);

	std::vector<int> ClusterOnce(pcl::PointCloud<pcl::PointXYZ>::Ptr votes);

	void VotesClustering(pcl::CorrespondencesPtr corrs_ori, pcl::PointCloud<pcl::PointXYZ>::Ptr votes, std::vector<pcl::CorrespondencesPtr, Eigen::aligned_allocator<pcl::CorrespondencesPtr>> &corrs_clusters);

	bool IsOverlap( pcl::PointCloud<PointT>::Ptr cloud1, pcl::PointCloud<PointT>::Ptr cloud2);

	void AdjustNormals(pcl::PointCloud<PointT>::Ptr cloud, pcl::PointCloud<NormalT>::Ptr cloud_normals, const pcl::PointXYZ &origin);

	void GetCSHOTCorrs(const pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr model, const pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr scene, 
				pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr model_keypoints, pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr scene_keypoints, pcl::CorrespondencesPtr model_scene_corrs, float model_resolution);

	void GetCSHOTCorrs(pcl::PointCloud<Descriptor3DType>::Ptr model_descriptors, pcl::PointCloud<Descriptor3DType>::Ptr scene_descriptors, pcl::CorrespondencesPtr model_scene_corrs);
};