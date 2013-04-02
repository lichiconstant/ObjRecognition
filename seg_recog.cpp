#include "Recognizer.h"

int main (int argc, char *argv[])
{
	bool showFeatures = false;

	int color[16][3] = { {255,0,0}, {0,255,0}, {0,0,255}, {255,255,0}, {255,0,255}, {0,255,255} };

	float CSHOT_ratio = 25;
	float model_sample_rate = 3;
	float part_ratio = 2;
	float normalDiff = 1.8;
	int searchNeighs = 10;

	bool disable_SIFT = false;
	bool disable_SHOT = false;

	if (pcl::console::find_switch (argc, argv, "-dshot"))   //enable the plane estimation
		disable_SIFT = true;
	if (pcl::console::find_switch (argc, argv, "-dsift"))   //enable the plane estimation
		disable_SHOT = true;

    int index_s,index_e;
    std::string modelpath(argv[1]);		                //path for full model
	std::string partpath(argv[2]);
	pcl::console::parse_argument (argc, argv, "--r", model_sample_rate);
	pcl::console::parse_argument (argc, argv, "--nd", normalDiff);
	pcl::console::parse_argument (argc, argv, "--sn", searchNeighs);
	std::string partname, partname_n;

	if( sscanf( argv[3], "%d", &index_s ) != 1 ){   //starting index (integer)
        std::cerr<<"Please input m_spcount\n";
        return -1;
    }
    if( sscanf( argv[4], "%d", &index_e ) != 1 ){   //ending index (integer)
        std::cerr<<"Please input m_spcount\n";
        return -1;
    }
	pcl::visualization::PCLVisualizer::Ptr viewer(new pcl::visualization::PCLVisualizer("Scene"));
	viewer->initCameraParameters();
	pcl::visualization::PCLVisualizer::Ptr viewer1(new pcl::visualization::PCLVisualizer("Model"));
	viewer1->initCameraParameters();

	std::ifstream ifs(modelpath + "\\Models.txt", std::ios::in );
	std::vector<SegRecog, Eigen::aligned_allocator<SegRecog>> recog_vec;

	std::vector<std::string> modelName;

	while(!ifs.eof()){
		char model_name[64]={0};
		ifs.getline(model_name, 64);
		if( model_name[0] == '#')		//comment that model
			continue;
		std::string temp_name(model_name);
		modelName.push_back(temp_name);

		std::cerr<<"Loading: "<<modelpath+"\\"+model_name<<std::endl;

		SegRecog recognizer(modelpath+"\\"+model_name, model_sample_rate);
		recognizer.setNormalDiff(normalDiff);
		recognizer.setSearchNeighs(searchNeighs);
		if( disable_SIFT )
			recognizer.DisableSIFT();
		if (disable_SHOT )
			recognizer.DisableSHOT();

		recog_vec.push_back(recognizer);
	}
	ifs.close();
	
	/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	for( int i = index_s; i <= index_e; i++ )
	{
        std::ostringstream convert;     // stream used for the conversion
        convert << i;

		pcl::PointCloud<PointXYZRGBIM>::Ptr scene(new pcl::PointCloud<PointXYZRGBIM>());
		std::cerr<<(partname=partpath +"\\" + "Hackerman_" + convert.str()+"_f.pcd")<<std::endl;
		pcl::io::loadPCDFile(partname, *scene);
		pcl::PointCloud<PointT>::Ptr scene_rgb(new pcl::PointCloud<PointT>());
		pcl::copyPointCloud(*scene, *scene_rgb);
		viewer->removeAllPointClouds();
		viewer1->removeAllPointClouds();
		viewer->addPointCloud(scene_rgb, "scene");
		pcl::PointCloud<NormalT>::Ptr scene_normals(new pcl::PointCloud<NormalT>());
		std::cerr<<(partname_n=partpath +"\\" + "Hackerman_" + convert.str()+"_n.pcd")<<std::endl;
		pcl::io::loadPCDFile(partname_n, *scene_normals);

		viewer->removeAllPointClouds();
		viewer->addPointCloud(scene_rgb, "scene");

		for ( int j = 0 ; j < recog_vec.size() ; j++ )
		{
			std::ostringstream convert1;     // stream used for the conversion
			convert1 <<j;
			if( j == 0 )
				recog_vec[j].LoadScene(scene, scene_normals);
			else
				recog_vec[j].LoadScene(recog_vec[0].segs_rgb, recog_vec[0].segs_normals, recog_vec[0].segs_shot_rgb, recog_vec[0].segs_shot_normals, recog_vec[0].segs_SHOTDescr, recog_vec[0].segs_sift_rgb, recog_vec[0].segs_sift_normals, recog_vec[0].segs_SIFTDescr);
			recog_vec[j].Recognize();

			std::vector<Eigen::Matrix4f, Eigen::aligned_allocator<Eigen::Matrix4f>> recog_result = recog_vec[j].GetReocgResult();
		
			///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////		
			viewer->removeAllPointClouds();
			viewer->addPointCloud(scene_rgb, "scene");
			viewer1->removeAllPointClouds();

			for( int k = 0 ; k < recog_result.size() ; k++ )
			{
				std::ostringstream convert2;     // stream used for the conversion
				convert2<<k;
				if( recog_result[k] != Eigen::Matrix4f::Identity() )
				{
					std::cerr<<"Got it!"<<std::endl;
					pcl::PointCloud<PointT>::Ptr model_rgb = recog_vec[j].GetModelMesh();
					pcl::PointCloud<PointT>::Ptr trans_model(new pcl::PointCloud<PointT>());
					Eigen::Matrix4f trans = recog_result[k].inverse();
					pcl::transformPointCloud(*model_rgb, *trans_model, trans);
					viewer->addPointCloud(trans_model, "trans_model"+convert1.str()+convert2.str());
					viewer1->addPointCloud(trans_model, "trans_model"+convert1.str()+convert2.str());
					viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_COLOR, color[j][0], color[j][1], color[j][2], "trans_model"+convert1.str()+convert2.str());
					//viewer->spin();
				}
			}
			viewer->setCameraPosition(0.9,0.0,0.2, 0.0, 1.0, 0.0);
			viewer->saveScreenshot(partpath + "\\recog_result\\"+modelName[j]+"_recog_"+ convert.str() + "_0.png");
			viewer->setCameraPosition(-0.9,0.0,-0.2, 0.0, 1.0, 0.0);
			viewer->saveScreenshot(partpath + "\\recog_result\\"+modelName[j]+"_recog_"+ convert.str() + "_1.png");
			viewer->setCameraPosition(0.0,0.0,-0.2, 0.0, 1.0, 0.0);
			viewer->saveScreenshot(partpath + "\\recog_result\\"+modelName[j]+"_recog_"+ convert.str() + "_2.png");
			viewer1->saveScreenshot(partpath + "\\recog_result\\"+modelName[j]+"_recog_"+ convert.str() + "_3.png");
		}
		
	}	
	
	return 1;
}
