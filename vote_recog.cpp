#include "Recognizer.h"

std::string fullmodel_path("E:\\JHU3d\\Joel\\fullmodel\\");
std::string model_prefix("Ovaltine");
float sift_threshold(1);

int color[16][3] = { {255,0,0}, {0,255,0}, {0,0,255}, {255,255,0}, {255,0,255}, {0,255,255} };

int main(int argc, char **argv)
{
	int index_s,index_e, i, j, k;
    std::string inputpath(argv[1]);                 //path for raw scene data     
	std::string outputpath(argv[2]);                //path for raw scene data     

	float sift_ratio(0.7);
	float resolution(0.00126);
	float bin_size(0.05);
	float curvature_threshold(0.03);
	int inlier_threshold(10);
	float sampling_rate(7), CSHOT_threshold(0.25);

	pcl::console::parse_argument (argc, argv, "--model_name", model_prefix);
	pcl::console::parse_argument (argc, argv, "--sift_ratio", sift_ratio);
	pcl::console::parse_argument (argc, argv, "--rf_ratio", rf_ratio);
	pcl::console::parse_argument (argc, argv, "--sift_threshold", sift_threshold);
	pcl::console::parse_argument (argc, argv, "--bin_size", bin_size);
	pcl::console::parse_argument (argc, argv, "--inlier_threshold", inlier_threshold);
	pcl::console::parse_argument (argc, argv, "--curvature_threshold", curvature_threshold);
	pcl::console::parse_argument (argc, argv, "--CSHOT_ratio", CSHOT_ratio);
	pcl::console::parse_argument (argc, argv, "--CSHOT_threshold", CSHOT_threshold);
	pcl::console::parse_argument (argc, argv, "--sampling_rate", sampling_rate);

	if( sscanf( argv[3], "%d", &index_s ) != 1 ){   //starting index (integer)
        std::cerr<<"Please input m_spcount\n";
        return -1;
    }
    if( sscanf( argv[4], "%d", &index_e ) != 1 ){   //ending index (integer)
        std::cerr<<"Please input m_spcount\n";
        return -1;
    }

	bool show_voting(false);
	bool show_surface(false);
	bool refined_keypoints(false);
	
	if (pcl::console::find_switch (argc, argv, "-sv"))
		show_voting = true;
	if (pcl::console::find_switch (argc, argv, "-ss"))
		show_surface = true;

	pcl::visualization::PCLVisualizer::Ptr viewer(new pcl::visualization::PCLVisualizer("Scene"));
	viewer->initCameraParameters();
	pcl::visualization::PCLVisualizer::Ptr viewer1(new pcl::visualization::PCLVisualizer("Model"));
	viewer1->initCameraParameters();

	///////////////////////////////////////////////////Loading Models///////////////////////////////////////////////////////////
	std::ifstream ifs(fullmodel_path + "Models.txt", std::ios::in );
	char buf[64];
	ifs.getline(buf, 64);
	if( buf[0] == '1' )
		refined_keypoints = true;
	else if( buf[0] == '0' )
		refined_keypoints = false;
	else
	{
		std::cerr<<"Input Model File Invalid!"<<std::endl;
		return 0;
	}
	std::vector<VotingRecog, Eigen::aligned_allocator<VotingRecog>> recog_vec;
	while(!ifs.eof()){
		char model_name[64];
		ifs.getline(model_name, 64);

		VotingRecog recog( fullmodel_path + model_name, refined_keypoints );
		recog.setBinSize(bin_size);
		recog.setNormalRatio(normal_ratio);
		recog.setCurvature(curvature_threshold);
		recog.setInlierNumber(inlier_threshold);
		recog.setResolution(resolution);
		recog.setRFRatio(rf_ratio);
		recog.setSIFTRatio(sift_ratio);
		recog.setShowVoting(show_voting);
		recog.setShowSurface(show_surface);
		recog.setPCLViewer(viewer);
		recog.setSamplingRate(sampling_rate);
		recog.setCSHOTRatio(CSHOT_ratio);
		recog.setCSHOTThreshold(CSHOT_threshold);

		recog_vec.push_back(recog);
	}
	ifs.close();
	/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

	for(i = index_s; i <= index_e; i++ )
	{
		std::ostringstream convert;     // stream used for the conversion
        convert << i;
		viewer->removeAllPointClouds();
		viewer->removeAllShapes();
		viewer1->removeAllPointClouds();
		viewer1->removeAllShapes();

		std::string filename = inputpath + "\\Hackerman_" + convert.str()+"_f.pcd";
		std::string filename_n = inputpath + "\\Hackerman_" + convert.str()+"_n.pcd";
		std::cerr<<filename<<std::endl;

		pcl::PointCloud<PointXYZRGBIM>::Ptr scene(new pcl::PointCloud<PointXYZRGBIM>);
		pcl::io::loadPCDFile(filename, *scene);
		pcl::PointCloud<NormalT>::Ptr scene_normals(new pcl::PointCloud<NormalT>);
		pcl::io::loadPCDFile(filename_n, *scene_normals);

		for( k = 0 ; k < recog_vec.size() ; k++ )
		{
			if( k == 0 )
			{
				recog_vec[k].LoadScene(scene, scene_normals);
				viewer->addPointCloud(recog_vec[k].scene_rgb, "scene");	
			}
			else
				recog_vec[k].LoadScene(recog_vec[0].scene_rgb, recog_vec[0].scene_normals, recog_vec[0].scene_keypoints, recog_vec[0].scene_keypoints_normals, recog_vec[0].scene_rf, recog_vec[0].scene_SIFTDescr, recog_vec[0].scene_CSHOTDescr);
			
			recog_vec[k].Recognize();

			std::cerr<<"Model "<<k<<" has "<<recog_vec[k].candidates.size()<<" candidates!"<<std::endl;
			for( j = 0 ; j < recog_vec[k].candidates.size() ; j++ ){
				std::ostringstream convert1;     // stream used for the conversion
				convert1 << k <<" "<< j;
				pcl::visualization::PointCloudColorHandlerCustom<PointT> candidate_handler (recog_vec[k].candidates[j], color[k][0], color[k][1], color[k][2]);
				viewer->addPointCloud(recog_vec[k].candidates[j], candidate_handler, "candidate" + convert1.str());
				viewer1->addPointCloud(recog_vec[k].candidates[j], convert1.str());
			}
		}
		
		//////////////////////////////////////Save Result/////////////////////////////////////////////////////////////////////////

		viewer->setCameraPosition(0.9,0.0,0.3, 0.0, 1.0, 0.0);
		viewer->saveScreenshot(outputpath + "\\recog_"+ convert.str() + "_0.png");
		viewer->setCameraPosition(-0.9,0.0,0.3, 0.0, 1.0, 0.0);
		viewer->saveScreenshot(outputpath + "\\recog_"+ convert.str() + "_1.png");
		viewer->setCameraPosition(0.0,0.0,-0.1, 0.0, 1.0, 0.0);
		viewer->saveScreenshot(outputpath + "\\recog_"+ convert.str() + "_2.png");
		
		viewer1->saveScreenshot(outputpath + "\\pos_"+ convert.str() + ".png");

		for( k = 0 ; k < recog_vec.size() ; k++ )
			recog_vec[k].Reset();
		
	}
	
	//viewer.addPointCloud(reduced_model_rgb, "reduced_model");
	return 1;
}