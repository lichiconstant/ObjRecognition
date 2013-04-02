#include "features.h"

#include <pcl/segmentation/sac_segmentation.h>
#include <pcl/segmentation/extract_clusters.h>
#include <pcl/filters/project_inliers.h>

int main (int argc, char *argv[])
{
    float planeT = 0.01;
    float zmax = 2.0;
    float zmin = 0.3;
    float x_left = -0.5;
    float x_right = 0.5;
    //float elevation = 0.055; //2
	float elevation_min = 0.055;  //1
	float elevation_max = 2;
	int cluster_size = 500;
    // the default plane parameters, it will be changed when enable the plane estimation
	//3
	float planea =  0.043893;    
    float planeb = -0.906755;
    float planec =  0.419368;
    float planed = -0.506173;

	//2    
	//float planea =  0.0216105;    
    //float planeb = -0.911367;
    //float planec =  0.411027;
    //float planed = -0.483404;
    
	//1
	//float planea =  0.0380406;    
    //float planeb = -0.870385;
    //float planec =  0.490901;
    //float planed = -0.707362;
    
	//dense
	//float planea =  0.027479;
	//float planeb = -0.88449;
	//float planec =  0.46575;
	//float planed = -0.433954;

	bool has_plane = false;
	bool stopped = false;
	bool EuclideanFilter = true;

    if (pcl::console::find_switch (argc, argv, "-p"))                       //enable the plane estimation
        has_plane = true;

	if (pcl::console::find_switch (argc, argv, "-stop"))                       //enable the plane estimation
        stopped = true;

	if (pcl::console::find_switch (argc, argv, "-e"))                       //enable the plane estimation
        EuclideanFilter = false;

    pcl::console::parse_argument (argc, argv, "--planeT", planeT);          //Threshold for excluding outliers while estimating the plane
    pcl::console::parse_argument (argc, argv, "--zmax", zmax);              //The maximum distance from Kinect that the object appears (interest of region)
    pcl::console::parse_argument (argc, argv, "--zmin", zmin);              //The minimum distance from Kinect of the interest region
    pcl::console::parse_argument (argc, argv, "--x_left", x_left);          //The left boundary coordinate of the interest region 
    pcl::console::parse_argument (argc, argv, "--x_right", x_right);        //The right boundary coordinate of the interest region 
    pcl::console::parse_argument (argc, argv, "--elevation_min", elevation_min);    //the object will appear above certain elevation beyond the ground plane
	pcl::console::parse_argument (argc, argv, "--elevation_max", elevation_max);    //the object will appear below certain elevation beyond the ground plane

    int index_s,index_e;
    std::string inputpath(argv[1]);                 //path for raw scene data                          
    std::string outputpath(argv[2]);                //path for output model data
    std::string scene_prefix("Hackerman_");         //prefix for raw scene data
    std::string model_prefix("Model_");             //prefix for model data
    std::cout<<argv[3]<<" "<<argv[4]<<std::endl;
    if( sscanf( argv[3], "%d", &index_s ) != 1 ){   //starting index (integer)
        std::cerr<<"Please input m_spcount\n";
        return -1;
    }
    if( sscanf( argv[4], "%d", &index_e ) != 1 ){   //ending index (integer)
        std::cerr<<"Please input m_spcount\n";
        return -1;
    }
    int i;
	pcl::visualization::PCLVisualizer viewer ("Object Model");
	viewer.addCoordinateSystem(0.3);
	int count = 0;
    for( i = index_s; i <= index_e; i++ , count++)
    {
        std::ostringstream convert;     // stream used for the conversion
        convert << i;

        pcl::PointCloud<PointXYZRGBIM>::Ptr scene_(new pcl::PointCloud<PointXYZRGBIM>); 
        pcl::PointCloud<PointXYZRGBIM>::Ptr scene(new pcl::PointCloud<PointXYZRGBIM>); 
        
		std::cerr<<inputpath +"\\" + scene_prefix + convert.str()+".pcd"<<std::endl;
        pcl::io::loadPCDFile(inputpath +"\\" + scene_prefix + convert.str()+".pcd", *scene_);
		int kk;

		cv::Mat image = cv::Mat::zeros(480, 640, CV_8UC3);
		for( kk = 0 ; kk < scene_->points.size(); kk++ )
		{
			uint32_t rgb = *reinterpret_cast<int*>(&scene_->points[kk].rgb);
                
			uint8_t r = (rgb >> 16) & 0x0000ff;
			uint8_t g = (rgb >> 8)  & 0x0000ff;
			uint8_t b = (rgb)       & 0x0000ff;
			
			int cc = scene_->points[kk].imX;
			int rr = scene_->points[kk].imY;
			image.at<uint8_t>(rr, cc*3+0) = b;
			image.at<uint8_t>(rr, cc*3+1) = g;
			image.at<uint8_t>(rr, cc*3+2) = r;
			if( scene_->points[kk].z > 0 )
				scene->points.push_back(scene_->points[kk]);
		}
		std::cerr<<"Original point number: "<<scene_->points.size()<<std::endl<<"Filtered point number: "<<scene->points.size()<<std::endl;
		//cv::namedWindow("Image");
		//cv::imshow("Image", image);
		//cv::waitKey(0);

        pcl::PointCloud<pcl::PointXYZRGB>::Ptr sceneXYZ (new pcl::PointCloud<pcl::PointXYZRGB>());
        pcl::PointCloud<pcl::PointXYZRGB>::Ptr sceneori (new pcl::PointCloud<pcl::PointXYZRGB>());
        pcl::PointCloud<pcl::PointXYZRGB>::Ptr scene_projected (new pcl::PointCloud<pcl::PointXYZRGB>());
        pcl::copyPointCloud(*scene,*sceneXYZ);
        pcl::copyPointCloud(*scene,*sceneori);

        pcl::ModelCoefficients::Ptr coefficients (new pcl::ModelCoefficients);
        if( has_plane )
        {
            coefficients->values.push_back(planea);
            coefficients->values.push_back(planeb);
            coefficients->values.push_back(planec);
            coefficients->values.push_back(planed);
        }
        else
        {
            pcl::PointIndices::Ptr inliers (new pcl::PointIndices);
            // Create the segmentation object
            pcl::SACSegmentation<pcl::PointXYZRGB> seg;
            // Optional
            seg.setOptimizeCoefficients (true);
            // Mandatory
            seg.setModelType (pcl::SACMODEL_PLANE);
            seg.setMethodType (pcl::SAC_RANSAC);
            seg.setDistanceThreshold (planeT);

            seg.setInputCloud (sceneXYZ);
            seg.segment (*inliers, *coefficients);
    
            if (inliers->indices.size () == 0)
            {
                PCL_ERROR ("Could not estimate a planar model for the given dataset.");
                return (-1);
            }

            std::cerr << "Model coefficients: " << coefficients->values[0] << " " 
                                                << coefficients->values[1] << " "
                                                << coefficients->values[2] << " " 
                                                << coefficients->values[3] << std::endl;

            std::cerr << "Model inliers: " << inliers->indices.size () << std::endl;
    
            //pcl::PointCloud<pcl::PointXYZRGB>::Ptr planepts (new pcl::PointCloud<pcl::PointXYZRGB>());
            //for (size_t i = 0; i < inliers->indices.size (); ++i)
            //    planepts->points.push_back(sceneXYZ->points[inliers->indices[i]]);
        }

        // Create the filtering object
        pcl::ProjectInliers<pcl::PointXYZRGB> proj;
        proj.setModelType (pcl::SACMODEL_PLANE);
        proj.setInputCloud (sceneori);
        proj.setModelCoefficients (coefficients);
        proj.filter (*scene_projected);

        std::cerr << "Cloud after projection: " << std::endl;
        pcl::PointCloud<PointXYZRGBIM>::Ptr model (new pcl::PointCloud<PointXYZRGBIM>());
        for (size_t i = 0; i < scene_projected->points.size (); ++i)
        {
            //std::cerr<<scene_projected->points[i].x << " "<< scene_projected->points[i].y << " " << scene_projected->points[i].z << std::endl;
            //distance from the point to the plane
            float dist = sqrt( (scene_projected->points[i].x-sceneori->points[i].x) * (scene_projected->points[i].x-sceneori->points[i].x) +
                               (scene_projected->points[i].y-sceneori->points[i].y) * (scene_projected->points[i].y-sceneori->points[i].y) +
                               (scene_projected->points[i].z-sceneori->points[i].z) * (scene_projected->points[i].z-sceneori->points[i].z) );

			if ( dist >= elevation_min && dist <= elevation_max  
                && scene_projected->points[i].z <= zmax 
                && scene_projected->points[i].z >= zmin 
                && scene_projected->points[i].x <= x_right
                && scene_projected->points[i].x >= x_left && scene_projected->points[i].z-sceneori->points[i].z >= 0)
                model->points.push_back(scene->points[i]);
        }
        model->width = model->points.size();
        model->height = 1;
        
		pcl::PointCloud<pcl::PointXYZRGB>::Ptr model_rgb (new pcl::PointCloud<pcl::PointXYZRGB>());
		pcl::copyPointCloud(*model, *model_rgb);

		pcl::PointCloud<PointXYZRGBIM>::Ptr model_filtered(new pcl::PointCloud<PointXYZRGBIM>);

		 // Creating the KdTree object for the search method of the extraction
		if( EuclideanFilter )
		{
			pcl::PointCloud<pcl::PointXYZ>::Ptr modelxyz (new pcl::PointCloud<pcl::PointXYZ>());
			pcl::copyPointCloud(*model,*modelxyz);
			pcl::search::KdTree<pcl::PointXYZ>::Ptr tree (new pcl::search::KdTree<pcl::PointXYZ>);
			tree->setInputCloud(modelxyz);

			std::vector<pcl::PointIndices> cluster_indices;
			pcl::EuclideanClusterExtraction<pcl::PointXYZ> ec;
			ec.setClusterTolerance (0.02); // 2cm
			ec.setMinClusterSize (cluster_size);
			ec.setMaxClusterSize (25000000);
			ec.setSearchMethod (tree);
			ec.setInputCloud (modelxyz);
			ec.extract (cluster_indices);

			std::cerr<<"Cluster Number: "<<cluster_indices.size()<<std::endl;

			int j = 0;
			for (std::vector<pcl::PointIndices>::const_iterator it = cluster_indices.begin (); it != cluster_indices.end (); ++it)
			{
				for (std::vector<int>::const_iterator pit = it->indices.begin (); pit != it->indices.end (); pit++)
					model_filtered->points.push_back (model->points[*pit]); 
			}

			std::cerr<<"Original point number: "<<model->points.size()<<std::endl<<"Filtered point number: "<<model_filtered->points.size()<<std::endl;
		}
		else
			pcl::copyPointCloud(*model, *model_filtered);

		pcl::PointCloud<pcl::PointXYZRGB>::Ptr model_filtered_rgb (new pcl::PointCloud<pcl::PointXYZRGB>());
        pcl::copyPointCloud(*model_filtered,*model_filtered_rgb);
        
		if( i == index_s )
			viewer.addPointCloud (model_filtered_rgb, "Model");
		else
			viewer.updatePointCloud (model_filtered_rgb, "Model");
		
		if( stopped == false )
			viewer.spinOnce(500);        //the duration for viewing is 1.5s
		else
			viewer.spin();
		pcl::io::savePCDFile(outputpath +"\\" + model_prefix + convert.str()+"_c.pcd", *model_filtered, true);
    }
    
    return 1;
}
