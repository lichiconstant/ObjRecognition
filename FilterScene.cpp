#include "features.h"

int main (int argc, char *argv[])
{
    int index_s,index_e;
    std::string inputpath(argv[1]);                 //path for raw scene data 
	std::string outputpath(argv[2]);                 //path for raw scene data
    std::string scene_prefix("Hackerman_");         //prefix for raw scene data
    
	if( sscanf( argv[3], "%d", &index_s ) != 1 ){   //starting index (integer)
        std::cerr<<"Please input m_spcount\n";
        return -1;
    }
    if( sscanf( argv[4], "%d", &index_e ) != 1 ){   //ending index (integer)
        std::cerr<<"Please input m_spcount\n";
        return -1;
    }
	bool show = false;
	
	if (pcl::console::find_switch (argc, argv, "-s"))   //enable the plane estimation
        show = true;

    int i,j;
	int count = 0;

    for( i = index_s; i <= index_e; i++ , count++)
    {
        std::ostringstream convert;			// stream used for the conversion
        convert << i;

        pcl::PointCloud<PointXYZRGBIM>::Ptr scene(new pcl::PointCloud<PointXYZRGBIM>); 
        
		std::cerr<<inputpath +"\\" + scene_prefix + convert.str()+".pcd"<<std::endl;
        pcl::io::loadPCDFile(inputpath +"\\" + scene_prefix + convert.str()+".pcd", *scene);
		pcl::PointCloud<PointXYZRGBIM>::Ptr filtered_scene (new pcl::PointCloud<PointXYZRGBIM>());

		for( j=0; j < scene->points.size() ; j++)
			if( scene->points[j].z > 0 )
				filtered_scene->points.push_back(scene->points[j]);
		filtered_scene->height = 1;
		filtered_scene->width = filtered_scene->points.size();

		pcl::PointCloud<PointT>::Ptr filtered_rgb(new pcl::PointCloud<PointT>());
		pcl::copyPointCloud(*filtered_scene, *filtered_rgb);
		float resolution = 0.0012;
		std::cerr<<resolution<<" "<<filtered_rgb->size()<<" "<<resolution*normal_ratio<<std::endl;
		pcl::PointCloud<NormalT>::Ptr filtered_normals(new pcl::PointCloud<NormalT>());
		computeNormals(filtered_rgb, filtered_normals, resolution*normal_ratio);
		
		pcl::io::savePCDFile(outputpath + "\\" + scene_prefix + convert.str()+ "_f.pcd", *filtered_scene, true);
		pcl::io::savePCDFile(outputpath + "\\" + scene_prefix + convert.str()+ "_n.pcd", *filtered_normals, true);

		if(show)
		{
			pcl::visualization::PCLVisualizer::Ptr viewer(new pcl::visualization::PCLVisualizer("Scene"));
			viewer->initCameraParameters();
			viewer->addPointCloudNormals<PointT, NormalT>(filtered_rgb, filtered_normals, 50, 0.02, "normals");
			viewer->spin();
		}
    }	
		
    return 1;
}
