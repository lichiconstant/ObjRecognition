#ifndef FEATURE_ESTIMATION_H
#define FEATURE_ESTIMATION_H

#include "typedefs.h"

#include <pcl/io/io.h>
#include <pcl/io/pcd_io.h>
#include <pcl/features/normal_3d_omp.h>
#include <pcl/features/shot_omp.h>
#include <pcl/keypoints/sift_keypoint.h>
#include <pcl/features/fpfh_omp.h>
#include <pcl/features/vfh.h>
#include <pcl/search/kdtree.h>
#include <pcl/search/search.h>

#include <pcl/point_cloud.h>
#include <pcl/correspondence.h>
#include <pcl/features/board.h>
#include <pcl/keypoints/uniform_sampling.h>
#include <pcl/recognition/cg/hough_3d.h>
#include <pcl/recognition/cg/geometric_consistency.h>
#include <pcl/visualization/pcl_visualizer.h>
//#include <pcl/kdtree/kdtree_flann.h>
//#include <pcl/kdtree/impl/kdtree_flann.hpp>
#include <pcl/common/transforms.h>
#include <pcl/console/parse.h>
#include <pcl/common/norms.h>
#include <pcl/registration/icp.h>
#include <pcl/registration/transformation_estimation_svd.h>
//#include <pcl/registration/transformation_estimation_lm.h>
#include <pcl/registration/correspondence_rejection_sample_consensus.h>
#include <pcl/registration/ia_ransac.h>

#include <opencv2\opencv.hpp>

struct PointXYZRGBIM
{
  union
  {
    struct
    {
      float x;
      float y;
      float z;
      float rgb;
      float imX;
      float imY;
    };
    float data[6];
  };
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
} EIGEN_ALIGN16;

POINT_CLOUD_REGISTER_POINT_STRUCT (PointXYZRGBIM,
                                    (float, x, x)
                                    (float, y, y)
                                    (float, z, z)
                                    (float, rgb, rgb)
                                    (float, imX, imX)
                                    (float, imY, imY)
)

struct PointObj
{
	union
	{
		float data[4];
		struct
		{
			float x;
			float y;
			float z;
		};
	};
	union
	{
		float data_n[4];
		float normal[3];
		struct
		{
			float normal_x;
			float normal_y;
			float normal_z;
		};
	};
	union
	{
		struct
		{
			float rgb;
			float curvature;
		};
		float data_c[4];
	};
	union
	{
		float data_s[4];
		float shift[3];
		struct
		{
			float shift_x;
			float shift_y;
			float shift_z;
		};
	};
	union
	{
        struct
        {
          float x_axis[3];
          float y_axis[3];
          float z_axis[3];
        };
        float rf[12];
	};
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
} EIGEN_ALIGN16;

POINT_CLOUD_REGISTER_POINT_STRUCT (PointObj,
                                    (float, x, x)
                                    (float, y, y)
                                    (float, z, z)
                                    (float, rgb, rgb)
                                    (float, curvature, curvature)
                                    (float, normal_x, normal_x)
									(float, normal_y, normal_y)
									(float, normal_z, normal_z)
									(float, shift_x, shift_x)
									(float, shift_y, shift_y)
									(float, shift_z, shift_z)
									(float[3], x_axis, x_axis)
									(float[3], y_axis, y_axis)
									(float[3], z_axis, z_axis)
)

typedef PointXYZRGBIM myPointT;
typedef pcl::SHOT352 SHOTdescr;
typedef pcl::SHOT1344 Descriptor3DType;


//PCL_EXPORTS std::ostream& operator << (std::ostream& os, const SIFTDescr& p);

struct SIFTDescr{
	float siftDescr[128];
	//friend std::ostream& operator << (std::ostream& os, const SIFTDescr& p);

	EIGEN_MAKE_ALIGNED_OPERATOR_NEW
} EIGEN_ALIGN16;

POINT_CLOUD_REGISTER_POINT_STRUCT (SIFTDescr,
									(float[128], siftDescr, siftDescr)
)

#define EPS 0.0000000001
#define PI 3.14159265358979

#define HNUM 50
#define SNUM 20
#define INUM 10

struct hsi_hist
{
	float histogram[HNUM + SNUM + INUM];
};

static float normal_ratio(25), CSHOT_ratio(25), rf_ratio(25), fpfh_ratio(27);

//fpfh ratio should be larger than normal ratio

/* Use NormalEstimation to estimate a cloud's surface normals 
 * Inputs:
 *   input
 *     The input point cloud
 *   radius
 *     The size of the local neighborhood used to estimate the surface
 * Return: A pointer to a SurfaceNormals point cloud
 */
SurfaceNormalsPtr estimateSurfaceNormals (const PointCloudPtr & input, float radius);

/* Use SIFTKeypoint to detect a set of keypoints
 * Inputs:
 *   points
 *     The input point cloud
 *   normals
 *     The input surface normals
 *   min_scale
 *     The smallest scale in the difference-of-Gaussians (DoG) scale-space
 *   nr_octaves
 *     The number of times the scale doubles in the DoG scale-space
 *   nr_scales_per_octave
 *     The number of scales computed for each doubling
 *   min_contrast
 *     The minimum local contrast that must be present for a keypoint to be detected
 * Return: A pointer to a point cloud of keypoints
 */
PointCloudPtr detectKeypoints (const PointCloudPtr & points, const SurfaceNormalsPtr & normals,
                 float min_scale, int nr_octaves, int nr_scales_per_octave, float min_contrast);

/* Use FPFHEstimation to compute local feature descriptors around each keypoint
 * Inputs:
 *   points
 *     The input point cloud
 *   normals
 *     The input surface normals
 *   keypoints
 *     A cloud of keypoints specifying the positions at which the descriptors should be computed
 *   feature_radius
 *     The size of the neighborhood from which the local descriptors will be computed 
 * Return: A pointer to a LocalDescriptors (a cloud of LocalDescriptorT points)
 */

FPFHDescriptorsPtr computeFPFHDescriptors (const PointCloudPtr & points, const SurfaceNormalsPtr & normals, 
                        const PointCloudPtr & keypoints, float feature_radius);

SHOTDescriptorsPtr computeSHOTDescriptors (const PointCloudPtr & points, const SurfaceNormalsPtr & normals, 
                        const PointCloudPtr & keypoints, float feature_radius);
/* Use VFHEstimation to compute a single global descriptor for the entire input cloud
 * Inputs:
 *   points
 *     The input point cloud
 *   normals
 *     The input surface normals
 * Return: A pointer to a GlobalDescriptors point cloud (a cloud containing a single GlobalDescriptorT point)
 */
GlobalDescriptorsPtr computeGlobalDescriptor (const PointCloudPtr & points, const SurfaceNormalsPtr & normals);

/* A simple structure for storing all of a cloud's features */
struct ObjectFeatures
{
  PointCloudPtr points;
  SurfaceNormalsPtr normals;
  PointCloudPtr keypoints;
  FPFHDescriptorsPtr FPFH_descriptors;
  SHOTDescriptorsPtr SHOT_descriptors;
  GlobalDescriptorsPtr global_descriptor;
};

/* Estimate normals, detect keypoints, and compute local and global descriptors 
 * Return: An ObjectFeatures struct containing all the features
 */
ObjectFeatures computeFeatures (const PointCloudPtr & input);

bool ComputeSIFT(pcl::PointCloud<myPointT>::Ptr mycloud, cv::Mat& cloud_image, cv::Mat& cloud_2DTo3D, 
                 std::vector<cv::KeyPoint>& cloud_sift_keypoints, cv::Mat& cloud_sift_descriptors);

void Find_SIFT_Corrs(std::vector< cv::DMatch >& cv_matches, pcl::CorrespondencesPtr model_scene_corrs, 
                     pcl::PointCloud<PointT>::Ptr cloud1, pcl::PointCloud<PointT>::Ptr cloud2,
                     cv::Mat cloud1_2DTo3D, cv::Mat cloud2_2DTo3D,
                     std::vector<cv::KeyPoint> cloud_sift_keypoints1, std::vector<cv::KeyPoint> cloud_sift_keypoints2,
                     cv::Mat cloud_sift_descriptors1, cv::Mat cloud_sift_descriptors2,
                     pcl::PointCloud<PointT>::Ptr cloud_keypoints1, pcl::PointCloud<PointT>::Ptr cloud_keypoints2, float sift_ratio, float *sim_score, bool *flags1, bool *flags2, std::vector< cv::DMatch >& cv_matches_show);

void SIFT_Matches(std::vector< cv::DMatch >& cv_matches, std::vector< cv::DMatch >& cv_matches_show, pcl::CorrespondencesPtr model_scene_corrs, 
                     pcl::PointCloud<PointT>::Ptr model_rgb, pcl::PointCloud<PointT>::Ptr scene_rgb,
                     cv::Mat model_2DTo3D, cv::Mat scene_2DTo3D,
                     std::vector<cv::KeyPoint> model_sift_keypoints, std::vector<cv::KeyPoint> scene_sift_keypoints,
                     cv::Mat model_sift_descriptors, cv::Mat scene_sift_descriptors,
                     pcl::PointCloud<PointT>::Ptr model_keypoints, pcl::PointCloud<PointT>::Ptr scene_keypoints, float sift_ratio);

void EasySIFT(pcl::PointCloud<myPointT>::Ptr cloud, pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr scene_temp_sift, pcl::PointCloud<SIFTDescr>::Ptr scene_SIFTDescr, std::vector<int> &sift_idx);

std::vector<int> FilterSIFT(pcl::PointCloud<PointXYZRGBIM>::Ptr model, pcl::PointCloud<NormalT>::Ptr model_normals, std::vector<cv::KeyPoint> &model_sift_keypoints, cv::Mat model_2DTo3D, cv::Mat pre_sift_descriptors, 
		pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr sift_cloud, cv::Mat &cur_sift_descriptors);

void RGBToHSI(int rgb[], float hsi[]);

double computeCloudResolution (const pcl::PointCloud<PointT>::ConstPtr &cloud);

void ComputeCentroid(pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud, pcl::PointCloud<pcl::PointXYZ>::Ptr center_cloud);

bool PreProcess(pcl::PointCloud<PointXYZRGBIM>::Ptr model, pcl::PointCloud<pcl::Normal>::Ptr model_normals, cv::Mat& model_image,
				cv::Mat& model_2DTo3D, std::vector<cv::KeyPoint>& model_sift_keypoints, cv::Mat& model_sift_descriptors, pcl::PointCloud<pcl::PointXYZ>::Ptr centroid, float& model_resolution);

void computeLocalFeatures(pcl::PointCloud<PointT>::Ptr cloud, pcl::PointCloud<PointT>::Ptr surface, pcl::PointCloud<NormalT>::Ptr surface_normals, pcl::PointCloud<pcl::FPFHSignature33>::Ptr features, float resolution);

void computeNormals(pcl::PointCloud<PointT>::Ptr cloud, pcl::PointCloud<NormalT>::Ptr cloud_normals, float normal_ss);

void computeKeyNormals(pcl::PointCloud<PointT>::Ptr keypoints, pcl::PointCloud<NormalT>::Ptr keypoints_normals, pcl::PointCloud<PointT>::Ptr surface, float normal_ss);

void RefineCloud(pcl::PointCloud<PointT>::Ptr cloud, pcl::PointCloud<PointT>::Ptr refined_cloud, float radius);

//void GetHueDominant(pcl::PointCloud<PointT>::Ptr model, pcl::PointCloud<PointT>::Ptr model_icp, pcl::PointCloud<PointT>::Ptr scene, pcl::PointCloud<PointT>::Ptr scene_icp);

void ExtractHue(pcl::PointCloud<PointT>::Ptr cloud, pcl::PointCloud<pcl::PointXYZHSV>::Ptr cloud_hue);

//void CloudHSI(pcl::PointCloud<pcl::PointXYZHSV>::Ptr cloud_hue, pcl::PointCloud<hsi_hist>::Ptr cloud_hsi, float cloud_resolution);

float SimHist(float *model_hist, float *scene_hist, int LEN);

//void AddCloud(pcl::PointCloud<PointObj>::Ptr original, pcl::PointCloud<PointObj>::Ptr cur);

//void AddNormal(pcl::PointCloud<NormalT>::Ptr original, pcl::PointCloud<NormalT>::Ptr cur);

std::vector<int> GetRawSIFT(pcl::PointCloud<PointXYZRGBIM>::Ptr model, pcl::PointCloud<NormalT>::Ptr model_normals, std::vector<cv::KeyPoint> &model_sift_keypoints, cv::Mat model_2DTo3D, cv::Mat pre_sift_descriptors, 
		pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr sift_cloud, cv::Mat &cur_sift_descriptors);

//void AddSIFTDescr(cv::Mat &ori_pool, const cv::Mat &newDescr);

//void AdjustFinalNormals(pcl::PointCloud<PointObj>::Ptr, const pcl::PointXYZ &origin);

void AdjustNormals(pcl::PointCloud<PointT>::Ptr cloud, pcl::PointCloud<NormalT>::Ptr cloud_normals, const pcl::PointXYZ &origin);

void ComputeCloudRF(pcl::PointCloud<PointT>::Ptr keypoints, pcl::PointCloud<NormalT>::Ptr surface_normals, 
	pcl::PointCloud<PointT>::Ptr surface, pcl::PointCloud<pcl::ReferenceFrame>::Ptr keypoints_rf, float rf_rad);

void ComputeShift(pcl::PointCloud<PointT>::Ptr keypoints, pcl::PointCloud<pcl::ReferenceFrame>::Ptr rf, pcl::PointCloud<pcl::PointXYZ>::Ptr shift, pcl::PointCloud<pcl::PointXYZ>::Ptr centroid);

void Voting(pcl::PointCloud<PointT>::Ptr scene_keypoints, pcl::PointCloud<pcl::ReferenceFrame>::Ptr scene_rf, pcl::PointCloud<pcl::PointXYZ>::Ptr model_shift, pcl::PointCloud<pcl::PointXYZ>::Ptr scene_shift);

void EasySIFT(pcl::PointCloud<myPointT>::Ptr cloud, pcl::PointCloud<NormalT>::Ptr cloud_normals, pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr cloud_temp_sift, pcl::PointCloud<SIFTDescr>::Ptr cloud_SIFTDescr, std::vector<int> &sift_idx);

std::vector<int> FilterSIFT(pcl::PointCloud<PointXYZRGBIM>::Ptr model, pcl::PointCloud<NormalT>::Ptr model_normals, std::vector<cv::KeyPoint> &model_sift_keypoints, cv::Mat model_2DTo3D, cv::Mat pre_sift_descriptors, 
		pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr sift_cloud, cv::Mat &cur_sift_descriptors);

#endif
