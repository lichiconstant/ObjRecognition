#include "features.h"

int main (int argc, char *argv[])
{
	bool showFeatures = false;
	if (pcl::console::find_switch (argc, argv, "-f"))   //enable the plane estimation
        showFeatures = true;

    int index_s,index_e;
    std::string modelname(argv[1]);		                //path for raw scene data    

	pcl::visualization::PCLVisualizer::Ptr viewer(new pcl::visualization::PCLVisualizer("Scene"));
	viewer->initCameraParameters();

	pcl::PointCloud<PointObj>::Ptr model(new pcl::PointCloud<PointObj>);
	pcl::io::loadPCDFile(modelname + "_fullmodel.pcd", *model);
    pcl::PointCloud<PointT>::Ptr model_rgb(new pcl::PointCloud<PointT>());
	pcl::copyPointCloud(*model, *model_rgb);
	
	float resolution;
	cv::FileStorage fs(modelname+"_aux.xml", cv::FileStorage::READ);
	fs["Resolution"] >> resolution;
	fs.release();
	std::cerr<<"Resolution: "<<resolution<<std::endl;

	pcl::PointCloud<PointT>::Ptr reduced_model_rgb (new pcl::PointCloud<PointT>());
	pcl::VoxelGrid<PointT> sor;
	sor.setInputCloud(model_rgb);
	sor.setLeafSize(resolution, resolution, resolution);
	sor.filter(*reduced_model_rgb);

	viewer->addPointCloud(reduced_model_rgb, "model");
	
	std::cerr<<reduced_model_rgb->points.size()<<std::endl;

	if( showFeatures )
	{
		pcl::PointCloud<PointObj>::Ptr model_keypoints(new pcl::PointCloud<PointObj>);
		pcl::PointCloud<PointT>::Ptr model_keypoints_rgb(new pcl::PointCloud<PointT>());

		std::cerr<<"Loading "<<modelname+"_siftkeypoints.pcd"<<std::endl;
		pcl::io::loadPCDFile(modelname+"_siftkeypoints.pcd", *model_keypoints);
		pcl::copyPointCloud(*model_keypoints, *model_keypoints_rgb);

		viewer->addPointCloud(model_keypoints_rgb, "keypoints");
		viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_COLOR, 255, 0, 0, "keypoints");
		viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 5, "keypoints");

		std::cerr<<model_keypoints_rgb->points.size()<<std::endl;
	}


	viewer->spin();
	return 1;
}
