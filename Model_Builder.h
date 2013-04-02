#include "features.h"
#include <pcl/filters/project_inliers.h>
#include <pcl\surface\mls.h>

//#define BSTEP 10

class Model_Builder{

public:
	Model_Builder();
	~Model_Builder();

	void Process(pcl::PointCloud<PointXYZRGBIM>::Ptr raw_model, pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr model, pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr model_keypoints, 
		pcl::PointCloud<SIFTDescr>::Ptr modelSIFTDescr, pcl::PointCloud<Descriptor3DType>::Ptr modelCSHOTDescr);

	void GenModel();

	void SaveModel(std::string modelname);

	std::vector<int> GetRawSIFT(pcl::PointCloud<PointXYZRGBIM>::Ptr model, pcl::PointCloud<NormalT>::Ptr model_normals, std::vector<cv::KeyPoint> &model_sift_keypoints, cv::Mat model_2DTo3D, cv::Mat pre_sift_descriptors, 
		pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr sift_cloud, cv::Mat &cur_sift_descriptors);

	void setICPThreshold(float icp_inlier_threshold_){icp_inlier_threshold = icp_inlier_threshold_;}				//0.07
	void setRANSACThreshold(float ransac_inlier_threshold_){ransac_inlier_threshold = ransac_inlier_threshold_;}	//0.01
	void setSIFTRatio(float sift_ratio_){sift_ratio = sift_ratio_;}						//0.7
	void setRFRatio(float rf_ratio_){rf_ratio = rf_ratio_;}								//25
	void setNormalRatio(float normal_ratio_){normal_ratio = normal_ratio_;}				//25
	void setKeyframeInliers(int keyframe_inliers_){keyframe_inliers = keyframe_inliers_;}			//30 or 40
	void setBLen(int Blen_){ BLen = Blen_; }
	void setSamplingRate(float sampling_rate_){ sampling_rate = sampling_rate_; }
	void setcropHeight(float cropHeight_){ cropHeight = cropHeight_; }
	void setCSHOTRatio(float CSHOT_ratio_){ CSHOT_ratio = CSHOT_ratio_; }
	void setCSHOTThreshold(float CSHOT_threshold_){ CSHOT_threshold = CSHOT_threshold_; }

	void setPlaneCoef(float planeCoef_[4])
	{
		memcpy(planeCoef, planeCoef_, sizeof(float)*4);
	}

	void setViewer(pcl::visualization::PCLVisualizer::Ptr viewer_){viewer = viewer_;}
	void setShowObject(bool show_object_){show_object = show_object_;}
	void setShowKeypoints(bool show_keypoints_){show_keypoints = show_keypoints_;}
	void setShowShift(bool show_shift_){show_shift = show_shift_;}
	void setShowRegion(bool show_region_){show_region = show_region_;}

	pcl::PointCloud<PointObj>::Ptr truemodel;
	pcl::PointCloud<PointObj>::Ptr trueSIFT;
	pcl::PointCloud<PointObj>::Ptr refined_trueSIFT;
	pcl::PointCloud<SIFTDescr>::Ptr final_sift;
	pcl::PointCloud<Descriptor3DType>::Ptr final_cshot;
	pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr final_mesh;
	cv::Mat sift_num;
	
private:
	
	pcl::PointCloud<PointObj>::Ptr fullmodel;				//raw point cloud
	pcl::PointCloud<PointObj>::Ptr fullmodel_SIFT;			//raw SIFT keypoints
	pcl::PointCloud<SIFTDescr>::Ptr SIFTDescr_pool;	//raw SIFT descriptors
	pcl::PointCloud<Descriptor3DType>::Ptr CSHOTDescr_pool;	//raw SIFT descriptors
	std::vector<int> sift_flag;								//seperate flag for each frame

	std::vector<Eigen::Matrix4f, Eigen::aligned_allocator<Eigen::Matrix4f> > transform_vec;
	Eigen::Matrix4f pre_initial_guess;
	std::vector<int> s_keyframe;

	std::vector<pcl::PointCloud<PointXYZRGBIM>::Ptr, Eigen::aligned_allocator<pcl::PointCloud<PointXYZRGBIM>::Ptr>> model_raw_vec;
	std::vector<pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr, Eigen::aligned_allocator<pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr>> model_vec;
	std::vector<pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr, Eigen::aligned_allocator<pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr>> model_keypoints_vec;
	std::vector<pcl::PointCloud<SIFTDescr>::Ptr, Eigen::aligned_allocator<pcl::PointCloud<SIFTDescr>::Ptr>> model_sift_vec;
	std::vector<pcl::PointCloud<Descriptor3DType>::Ptr, Eigen::aligned_allocator<pcl::PointCloud<Descriptor3DType>::Ptr>> model_cshot_vec;
	//std::vector<cv::Mat> model_sift_descriptors_vec;
	std::vector<float> model_resolution_vec;
	
	float icp_inlier_threshold;		//0.07
	float ransac_inlier_threshold;  //0.01
	float sift_ratio;				//0.7
	float rf_ratio;					//25
	float normal_ratio;				//25
	int keyframe_inliers;			//30 or 40
	float sampling_rate;
	float CSHOT_ratio;
	float CSHOT_threshold;
	float cropHeight;

	float planeCoef[4];

	bool show_object;
	bool show_keypoints;
	bool show_shift;
	bool show_region;

	int BLen;

	pcl::visualization::PCLVisualizer::Ptr viewer;

	bool TestKeyFrame(int test_id, int last_key_id, Eigen::Matrix4f &initial_guess); 

	void updateFullModel(pcl::PointCloud<PointObj>::Ptr curmodel, pcl::PointCloud<PointObj>::Ptr curSIFT, pcl::PointCloud<SIFTDescr>::Ptr modelSIFTDescr, pcl::PointCloud<Descriptor3DType>::Ptr modelCSHOTDescr);

	Eigen::Matrix4f DenseAlign(int model_id, int scene_id, Eigen::Matrix4f &initial_guess, pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr model_region);

	void FilterBoundary(pcl::PointCloud<PointXYZRGBIM>::Ptr model, pcl::PointCloud<PointT>::Ptr Filtered_region);   //fast boundary extractiong using 2D image

	std::vector<int> CropCloud(pcl::PointCloud<PointObj>::Ptr cloud, pcl::PointCloud<PointObj>::Ptr subcloud, float min, float max);

	bool myICP(pcl::PointCloud<PointT>::Ptr model, pcl::PointCloud<PointT>::Ptr scene, pcl::PointCloud<NormalT>::Ptr model_normals, pcl::PointCloud<NormalT>::Ptr scene_normals, 
		Eigen::Matrix4f& initial_guess, Eigen::Matrix4f& rotatranslation, float model_resolution, float scene_resolution);

	void Find_SIFT_Corrs(pcl::PointCloud<SIFTDescr>::Ptr descr1, pcl::PointCloud<SIFTDescr>::Ptr descr2, pcl::CorrespondencesPtr model_scene_corrs, bool *flag1, bool *flag2);

	void Match3DSIFT(pcl::PointCloud<SIFTDescr>::Ptr descr1, pcl::PointCloud<SIFTDescr>::Ptr descr2, pcl::CorrespondencesPtr model_scene_corrs);

	void AddCloud(pcl::PointCloud<PointObj>::Ptr original, pcl::PointCloud<PointObj>::Ptr cur);

	void AddSIFTDescr(cv::Mat &ori_pool, const cv::Mat &newDescr);

	void ExtractHue(pcl::PointCloud<PointT>::Ptr cloud, pcl::PointCloud<pcl::PointXYZHSV>::Ptr cloud_hue);

	void RGBToHSI(int rgb[], float hsi[]);

	double computeCloudResolution (const pcl::PointCloud<PointT>::ConstPtr &cloud);

	void ComputeCentroid(pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud, pcl::PointCloud<pcl::PointXYZ>::Ptr center_cloud);

	void ComputeCloudRF(pcl::PointCloud<PointT>::Ptr keypoints, const pcl::PointCloud<NormalT>::Ptr surface_normals, 
							const pcl::PointCloud<PointT>::Ptr surface, pcl::PointCloud<pcl::ReferenceFrame>::Ptr keypoints_rf, float rf_rad);

	void ComputeShift(pcl::PointCloud<PointT>::Ptr keypoints, pcl::PointCloud<pcl::ReferenceFrame>::Ptr rf, 
			pcl::PointCloud<pcl::PointXYZ>::Ptr shift, pcl::PointCloud<pcl::PointXYZ>::Ptr centroid);

	void computeNormals(pcl::PointCloud<PointT>::Ptr cloud, pcl::PointCloud<NormalT>::Ptr cloud_normals, float normal_ss);

	void computeKeyNormals(pcl::PointCloud<PointT>::Ptr keypoints, pcl::PointCloud<NormalT>::Ptr keypoints_normals, pcl::PointCloud<PointT>::Ptr surface, float normal_ss);

	void AdjustNormals(pcl::PointCloud<PointT>::Ptr cloud, pcl::PointCloud<NormalT>::Ptr cloud_normals, const pcl::PointXYZ &origin);

	void AdjustFinalNormals(pcl::PointCloud<PointObj>::Ptr, const pcl::PointXYZ &origin);

	void GetCSHOTCorrs(const pcl::PointCloud<Descriptor3DType>::Ptr model_descriptors, const pcl::PointCloud<Descriptor3DType>::Ptr scene_descriptors, pcl::CorrespondencesPtr model_scene_corrs);
};