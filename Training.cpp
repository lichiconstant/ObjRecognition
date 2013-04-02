#include "features.h"
#include <pcl/segmentation/region_growing.h>
#include <pcl/filters/extract_indices.h>
//#include <pcl\surface\mls.h>

#define ZMAX 2

int main(int argc, char *argv[])
{
	int index_s,index_e;
    std::string scenepath(argv[1]);		                //path for scene data
	if( sscanf( argv[2], "%d", &index_s ) != 1 ){   //starting index (integer)
        std::cerr<<"Please input m_spcount\n";
        return -1;
    }
    if( sscanf( argv[3], "%d", &index_e ) != 1 ){   //ending index (integer)
        std::cerr<<"Please input m_spcount\n";
        return -1;
    }

	float normalDiff = 1.8;
	int searchNeighs = 10;
	int minsize = 500;

	pcl::console::parse_argument (argc, argv, "--nd", normalDiff);
	pcl::console::parse_argument (argc, argv, "--sn", searchNeighs);
	pcl::console::parse_argument (argc, argv, "--msize", minsize);

	int width = 640;
	int height = 480;

	pcl::visualization::PCLVisualizer::Ptr viewer(new pcl::visualization::PCLVisualizer("Scene"));
	viewer->initCameraParameters();

	
	
	for( int i = index_s; i <= index_e; i++ )
	{
		std::string scene_c;
		std::string scene_n;
		std::ostringstream convert;     // stream used for the conversion
        convert << i;

		pcl::PointCloud<PointXYZRGBIM>::Ptr scene(new pcl::PointCloud<PointXYZRGBIM>());
		std::cerr<<(scene_c=scenepath +"\\" + "Hackerman_" + convert.str()+"_f.pcd")<<std::endl;
		pcl::io::loadPCDFile(scene_c, *scene);
		pcl::PointCloud<PointT>::Ptr scene_rgb(new pcl::PointCloud<PointT>());
		pcl::copyPointCloud(*scene, *scene_rgb);
		
		pcl::PointCloud<NormalT>::Ptr scene_normals(new pcl::PointCloud<NormalT>());
		std::cerr<<(scene_n=scenepath +"\\" + "Hackerman_" + convert.str()+"_n.pcd")<<std::endl;
		pcl::io::loadPCDFile(scene_n, *scene_normals);

		clock_t t1, t2;
		t1 = clock();
		pcl::search::KdTree<PointT>::Ptr tree(new pcl::search::KdTree<PointT>());
	
		pcl::RegionGrowing<PointT, NormalT> reg;
		reg.setMinClusterSize(1);
		reg.setMaxClusterSize(1000000);
		reg.setNumberOfNeighbours(searchNeighs);
		reg.setSearchMethod(tree);
		reg.setInputCloud(scene_rgb);
		reg.setInputNormals(scene_normals);
		
		reg.setSmoothnessThreshold (normalDiff / 180.0 * M_PI);
		reg.setCurvatureTestFlag(false);
		reg.setCurvatureThreshold (1.0);
	
		std::vector <pcl::PointIndices> clusters;
		reg.extract (clusters);
		t2 = clock();
		std::cerr<<"Segmentation Time Elapsed: "<<((double)t2-(double)t1)/CLOCKS_PER_SEC<<std::endl;
		std::cerr<<"Cluster Number: "<<clusters.size()<<std::endl;

		pcl::ExtractIndices<PointT> extract_rgb;
		extract_rgb.setInputCloud(scene_rgb);
		extract_rgb.setNegative(false);
		std::vector<pcl::PointCloud<PointT>::Ptr, Eigen::aligned_allocator<pcl::PointCloud<PointT>::Ptr>> main_segs_rgb;
		std::vector<pcl::PointCloud<PointT>::Ptr, Eigen::aligned_allocator<pcl::PointCloud<PointT>::Ptr>> sub_segs_rgb;
		std::vector<PointT, Eigen::aligned_allocator<PointT>> sub_center;

		pcl::PointCloud<PointT>::Ptr main_cloud(new pcl::PointCloud<PointT>());
		std::vector<int> labels;
		
		for( int j = 0, label_count = 0 ; j < clusters.size() ; j++ )
		{
			pcl::PointIndices::Ptr clusterPt(new pcl::PointIndices());
			clusterPt->indices = clusters[j].indices;
			clusterPt->header = clusters[j].header;
			extract_rgb.setIndices(clusterPt);
			pcl::PointCloud<PointT>::Ptr temp_rgb(new pcl::PointCloud<PointT>());
			extract_rgb.filter(*temp_rgb);

			if( temp_rgb->size() >= minsize )
			{
				main_segs_rgb.push_back(temp_rgb);
				main_cloud->insert(main_cloud->end(), temp_rgb->begin(), temp_rgb->end());
				labels.insert(labels.end(), temp_rgb->size(), label_count);
				//for( int k = 0 ; k < temp_rgb->size(); k++ )
				//{
				//	main_cloud->push_back(temp_rgb->at(k));
				//	labels.push_back(label_count);
				//}
				label_count++;
			}
			else
			{
				sub_segs_rgb.push_back(temp_rgb);
				pcl::PointCloud<pcl::PointXYZ>::Ptr center(new pcl::PointCloud<pcl::PointXYZ>());
				ComputeCentroid(temp_rgb, center);
				PointT buf;
				buf.x = center->at(0).x;
				buf.y = center->at(0).y;
				buf.z = center->at(0).z;
				pcl::search::KdTree<PointT> temp_tree;
				temp_tree.setInputCloud(temp_rgb);

				std::vector<int> temp_ind;
				std::vector<float> temp_dist;
				if( temp_tree.nearestKSearch(buf, 1, temp_ind, temp_dist) > 0 )
				{
					PointT ref_center(temp_rgb->at(temp_ind[0]));
					sub_center.push_back(ref_center);
				}
				else
				{
					std::cerr<<"Fatal Error..."<<std::endl;
					return 0;
				}
			}
		}
		std::cerr<<"main segments: "<<main_segs_rgb.size()<<std::endl;
		std::cerr<<"sub segments: "<<sub_segs_rgb.size()<<std::endl;

		pcl::search::KdTree<PointT> main_tree;
		main_tree.setInputCloud(main_cloud);
		for(int j = 0 ; j < sub_center.size() ; j++ )
		{
			std::vector<int> temp_ind;
			std::vector<float> temp_dist;
			if( main_tree.nearestKSearch(sub_center[j], 1, temp_ind, temp_dist) > 0 )
			{
				int merged_idx = labels[temp_ind[0]];
				main_segs_rgb.at(merged_idx)->insert(main_segs_rgb.at(merged_idx)->end(), sub_segs_rgb.at(j)->begin(), sub_segs_rgb.at(j)->end());
			}
		}
		
		std::srand(time(NULL));
		for( int j = 0 ; j < main_segs_rgb.size() ; j++ )
		{
			std::ostringstream convert1;     // stream used for the conversion
			convert1 << j;

			double r = rand()%255 / 255.0;
			double g = rand()%255 / 255.0;
			double b = rand()%255 / 255.0;
			viewer->addPointCloud(main_segs_rgb[j], "seg_result"+convert1.str());
			std::cerr<<r<<" "<<g<<" "<<b<<std::endl;
			viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_COLOR, r,g,b, "seg_result"+convert1.str());
		}
		//viewer->removeAllPointClouds();
		
		viewer->spin();

		/*
		cv::Mat color = cv::Mat::zeros(height, width, CV_8UC3);
		cv::Mat raw_depth = cv::Mat::zeros(height, width, CV_32FC1);
		cv::Mat raw_normals = cv::Mat::zeros(height, width, CV_32FC3);
		//cv::Mat depth = cv::Mat::zeros(height, width, CV_8UC1);
		cv::Mat normal_d = cv::Mat::zeros(height, width, CV_8UC1);
		cv::Mat grad_ori = cv::Mat::zeros(height, width, CV_8UC1);
		cv::Mat normals = cv::Mat::zeros(height, width, CV_8UC3);

		float dmax = 0;
		float dmin = 100;
		std::cerr<<"Dense Normals? "<<scene_normals->is_dense<<std::endl;
		for( int j = 0 ; j < scene_rgb->size() ; j++ )
		{
			if (scene_rgb->points[j].z < ZMAX)
			{
				int pt_x = scene->points[j].imX;
				int pt_y = scene->points[j].imY;

				color.at<unsigned char>(pt_y, pt_x*3+0) = scene_rgb->points[j].b;
				color.at<unsigned char>(pt_y, pt_x*3+1) = scene_rgb->points[j].g;
				color.at<unsigned char>(pt_y, pt_x*3+2) = scene_rgb->points[j].r;

				normals.at<unsigned char>(pt_y, pt_x*3+0) = (int)(fabs(scene_normals->points[j].normal_x) * 255);
				normals.at<unsigned char>(pt_y, pt_x*3+1) = (int)(fabs(scene_normals->points[j].normal_y) * 255);
				normals.at<unsigned char>(pt_y, pt_x*3+2) = (int)(fabs(scene_normals->points[j].normal_z) * 255);

				raw_normals.at<unsigned char>(pt_y, pt_x*3+0) = scene_normals->points[j].normal_x;
				raw_normals.at<unsigned char>(pt_y, pt_x*3+1) = scene_normals->points[j].normal_y;
				raw_normals.at<unsigned char>(pt_y, pt_x*3+2) = scene_normals->points[j].normal_z;

				
				if( dmax < scene_rgb->points[j].z)
					dmax = scene_rgb->points[j].z;
				if( dmin > scene_rgb->points[j].z)
					dmin = scene_rgb->points[j].z;
				
				raw_depth.at<float>(pt_y, pt_x) = scene_rgb->points[j].z;
			}
		}
		*/
		/*
		float diff = dmax - dmin;
		for( int j = 0 ; j < scene_rgb->size() ; j++ )
		{
			if (scene_rgb->points[j].z < ZMAX)
			{
				int pt_x = scene->points[j].imX;
				int pt_y = scene->points[j].imY;

				int depth_val = (scene_rgb->points[j].z - dmin)/diff * 255;

				depth.at<unsigned char>(pt_y, pt_x) = depth_val;
				
			}
		}
		*/
		/*
		for( int i = 1 ; i < height-1; i++ )
			for( int j = 1 ; j < width-1 ; j++ )
			{
				if( raw_depth.at<float>(i, j) != 0 )
				{
					float left_depth = raw_depth.at<float>(i, j-1) == 0 ? raw_depth.at<float>(i, j) : raw_depth.at<float>(i, j-1);
					float right_depth = raw_depth.at<float>(i, j+1) == 0 ? raw_depth.at<float>(i, j) : raw_depth.at<float>(i, j+1);
					float hor_diff = right_depth - left_depth;
					float up_depth = raw_depth.at<float>(i-1, j) == 0 ? raw_depth.at<float>(i, j) : raw_depth.at<float>(i-1, j);
					float down_depth = raw_depth.at<float>(i+1, j) == 0 ? raw_depth.at<float>(i, j) : raw_depth.at<float>(i+1, j);
					float ver_diff = up_depth - down_depth;

					float theta = atan2(ver_diff+0.0, hor_diff+0.0)/M_PI * 180;
					theta = theta < 0 ? theta + 360 : theta;
					grad_ori.at<unsigned char>(i, j) = int(255 * theta/360);
				}
			}

		cv::namedWindow("color");
		cv::imshow("color", color);
		//cv::namedWindow("depth");
		//cv::imshow("depth", depth);
		cv::namedWindow("grad_ori");
		cv::imshow("grad_ori", grad_ori);
		cv::namedWindow("normals");
		cv::imshow("normals", normals);
		cv::waitKey(0);

		cv::imwrite(scenepath +"\\color_"+convert.str()+".png", color);
		//cv::imwrite(scenepath +"\\depth_"+convert.str()+".png", depth);
		cv::imwrite(scenepath +"\\grad_ori_"+convert.str()+".png", grad_ori);
		cv::imwrite(scenepath +"\\normals_"+convert.str()+".png", normals);
		*/
	}
	return 1;
}



/*
int main (int argc, char *argv[])
{
    int index_s,index_e;
    std::string modelname(argv[1]);		                //path for full model

	bool show = false;
	if (pcl::console::find_switch (argc, argv, "-s"))   //enable the plane estimation
        show = true;

	pcl::visualization::PCLVisualizer::Ptr viewer(new pcl::visualization::PCLVisualizer("Scene"));
	viewer->initCameraParameters();
	
	float resolution;
	cv::FileStorage fs(modelname+"_aux.xml", cv::FileStorage::READ);
	fs["Resolution"] >> resolution;
	fs.release();
	std::cerr<<"Resolution: "<<resolution<<std::endl;

	pcl::PointCloud<PointObj>::Ptr model(new pcl::PointCloud<PointObj>);
	pcl::io::loadPCDFile(modelname + "_fullmodel.pcd", *model);
    pcl::PointCloud<PointT>::Ptr model_rgb(new pcl::PointCloud<PointT>());
	pcl::copyPointCloud(*model, *model_rgb);

	pcl::PointCloud<PointT>::Ptr reduced_model_rgb (new pcl::PointCloud<PointT>());
	pcl::VoxelGrid<PointT> sor;
	sor.setInputCloud(model_rgb);
	sor.setLeafSize(resolution, resolution, resolution);
	sor.filter(*reduced_model_rgb);

	// Create a KD-Tree
	pcl::search::KdTree<PointT>::Ptr tree (new pcl::search::KdTree<PointT>);
	pcl::PointCloud<PointT> mls_points;
	pcl::MovingLeastSquares<PointT, PointT> mls;
	// Set parameters
	mls.setInputCloud (reduced_model_rgb);
	mls.setComputeNormals(false);
	mls.setPolynomialFit(true);
	mls.setSearchMethod (tree);
	mls.setSearchRadius (0.01);
	// Reconstruct
	mls.process (mls_points);
	pcl::copyPointCloud(mls_points, *reduced_model_rgb);

	pcl::PointCloud<pcl::PointXYZ>::Ptr centroid(new pcl::PointCloud<pcl::PointXYZ>());;
	ComputeCentroid(reduced_model_rgb, centroid);

	pcl::PointCloud<NormalT>::Ptr reduced_model_normals (new pcl::PointCloud<NormalT>());
	computeNormals(reduced_model_rgb, reduced_model_normals, resolution*normal_ratio);
	AdjustNormals(reduced_model_rgb, reduced_model_normals, centroid->points[0]);
	std::cerr<<"FullModel Normal Computation Completed!"<<std::endl;
	
	pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr result(new pcl::PointCloud<pcl::PointXYZRGBNormal>());
	pcl::copyPointCloud(*reduced_model_rgb, *result);
	pcl::copyPointCloud(*reduced_model_normals, *result);

	pcl::io::savePCDFile(modelname+"_reduced.pcd", *result, true);

	if( show)
	{
		pcl::visualization::PCLVisualizer::Ptr viewer(new pcl::visualization::PCLVisualizer("Scene"));
		viewer->initCameraParameters();
		viewer->addPointCloudNormals<PointT, NormalT>(reduced_model_rgb, reduced_model_normals, 50, 0.02, "normals");
		viewer->spin();
	}

	//pcl::io::savePCDFile(modelname"_rgb.pcd", *reduced_model_rgb, true);
	//pcl::io::savePCDFile("_normals.pcd", *reduced_model_normals, true);
	return 1;
}
*/
