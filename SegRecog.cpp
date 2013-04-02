#include "Recognizer.h"
#include <pcl/segmentation/region_growing.h>
#include <pcl/filters/extract_indices.h>

SegRecog::SegRecog(std::string modelname, float model_sample_ratio_)
{
	model_sample_ratio = model_sample_ratio_;
	part_sample_ratio = 2;
	SIFT_iter_ratio = 0.2;

	curveT = 0.03;					//0.03
	hueT = 0.08;					//0.08
	corrs_ratio = 0.5;				//0.5
	converge_epsilon = 0.000001;	//0.000001

	// For region HSI histogram
	histT = 0.2;			//0.3
	h_bin = 32;				//32
	s_bin = 16;				//16
	i_bin = 16;				//16
	
	sift_ratio = 0.7;		//0.7

	//For Segmentation
	minSegment = 500;		//500
	maxSegment = 100000;	//100000
	searchNeighs = 10;		//10
	normalDiff = 1.8;		//1.8

	sift_engine = true;
	shot_engine = true;

#if defined(DEBUG) || defined(SEGSHOW)
	pcl::visualization::PCLVisualizer::Ptr viewer_(new pcl::visualization::PCLVisualizer("Model"));
	viewer = viewer_;
	viewer->initCameraParameters();
#endif

	pcl::PointCloud<PointT>::Ptr model_rgb_(new pcl::PointCloud<PointT>());
	pcl::PointCloud<NormalT>::Ptr model_normals_(new pcl::PointCloud<NormalT>());
	pcl::PointCloud<PointT>::Ptr model_shot_(new pcl::PointCloud<PointT>());
	pcl::PointCloud<NormalT>::Ptr model_shot_normals_(new pcl::PointCloud<NormalT>());
	pcl::PointCloud<SHOTdescr>::Ptr model_SHOTDescr_(new pcl::PointCloud<SHOTdescr>());
	pcl::PointCloud<PointT>::Ptr model_sift_(new pcl::PointCloud<PointT>());
	pcl::PointCloud<NormalT>::Ptr model_sift_normals_(new pcl::PointCloud<NormalT>());
	pcl::PointCloud<SIFTDescr>::Ptr model_SIFTDescr_(new pcl::PointCloud<SIFTDescr>());

	model_rgb = model_rgb_;
	model_normals = model_normals_;
	model_shot = model_shot_;
	model_shot_normals = model_shot_normals_;
	model_SHOTDescr = model_SHOTDescr_;
	model_sift = model_sift_;
	model_sift_normals = model_sift_normals_;
	model_SIFTDescr = model_SIFTDescr_;
	
	// Read resolution
	cv::FileStorage fs(modelname+"_aux.xml", cv::FileStorage::READ);
	fs["Resolution"] >> resolution;
	fs.release();
	std::cerr<<"Resolution: "<<resolution<<std::endl;
	distT = resolution * 7;			//resolution * 7

	// Read Full model mesh
	pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr model(new pcl::PointCloud<pcl::PointXYZRGBNormal>());
	pcl::io::loadPCDFile(modelname + "_reduced.pcd", *model);
	pcl::copyPointCloud(*model, *model_rgb);
	pcl::copyPointCloud(*model, *model_normals);

	pcl::VoxelGrid<PointT> sor;
	sor.setInputCloud(model_rgb);
	sor.setLeafSize(resolution*model_sample_ratio, resolution*model_sample_ratio, resolution*model_sample_ratio);
	sor.filter(*model_shot);	

	computeKeyNormals(model_shot, model_shot_normals, model_rgb, resolution*normal_ratio);
	pcl::PointCloud<pcl::PointXYZ>::Ptr centroid(new pcl::PointCloud<pcl::PointXYZ>());;
	ComputeCentroid(model_shot, centroid);
	AdjustNormals(model_shot, model_shot_normals, centroid->points[0]);

	pcl::SHOTEstimationOMP<PointT, NormalT, SHOTdescr, pcl::ReferenceFrame> descr_est;
	descr_est.setRadiusSearch (resolution*CSHOT_ratio);
	descr_est.setNumberOfThreads(THREADNUM);
	descr_est.setInputCloud(model_shot);
	descr_est.setInputNormals(model_normals);
	descr_est.setSearchSurface(model_rgb);
	descr_est.compute (*model_SHOTDescr);
	std::cerr<<"SHOT Keypoints: "<<model_rgb->size()<<" "<<model_shot->size()<<std::endl;
	/////////////////////////Loading SIFT Keypoints////////////////////////////////////
	
	pcl::PointCloud<PointObj>::Ptr model_sift_ori(new pcl::PointCloud<PointObj>);
	std::cerr<<"Loading "<<modelname+"_siftkeypoints_f.pcd"<<std::endl;
	pcl::io::loadPCDFile(modelname+"_siftkeypoints_f.pcd", *model_sift_ori);
	pcl::copyPointCloud(*model_sift_ori, *model_sift);
	pcl::copyPointCloud(*model_sift_ori, *model_sift_normals);

	std::cerr<<"Loading "<<modelname+"_siftdescr.pcd"<<std::endl;
	pcl::io::loadPCDFile(modelname+"_siftdescr.pcd", *model_SIFTDescr);

	std::cerr<<modelname<<" Full Model Ready!"<<std::endl<<std::endl;

	///////////////////////////////////////////////////////////////////////////////////
#ifdef DEBUG
	viewer->addPointCloud(model_rgb, "model");
	//viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 7, "center");
	//viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_COLOR, 255, 255, 255, "model");
	//viewer->addPointCloudNormals<PointT, NormalT>(reduced_keys, reduced_keys_normals, 50, 0.02, "normals");
	//viewer->spin();
#endif

	// Initialize Parameters


}

void SegRecog::LoadScene(pcl::PointCloud<PointXYZRGBIM>::Ptr scene, pcl::PointCloud<NormalT>::Ptr scene_normals)
{
	segs_rgb.clear();
	segs_normals.clear();
	segs_shot_rgb.clear();
	segs_shot_normals.clear();
	segs_SHOTDescr.clear();
	segs_sift_rgb.clear();
	segs_sift_normals.clear();
	segs_SIFTDescr.clear();

	pcl::PointCloud<PointT>::Ptr scene_rgb(new pcl::PointCloud<PointT>());
	pcl::copyPointCloud(*scene, *scene_rgb);

	pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr scene_temp_sift(new pcl::PointCloud<pcl::PointXYZRGBNormal>());
	std::vector<int> sift_idx;
	pcl::PointCloud<SIFTDescr>::Ptr scene_SIFTDescr(new pcl::PointCloud<SIFTDescr>());
	EasySIFT(scene, scene_normals, scene_temp_sift, scene_SIFTDescr, sift_idx);

	pcl::PointCloud<PointT>::Ptr scene_sift(new pcl::PointCloud<PointT>);
	pcl::PointCloud<NormalT>::Ptr scene_sift_normals(new pcl::PointCloud<NormalT>());
	pcl::copyPointCloud(*scene_temp_sift, *scene_sift);
	pcl::copyPointCloud(*scene_temp_sift, *scene_sift_normals);

	extractSegments(scene_rgb, scene_normals, scene_sift, scene_sift_normals, scene_SIFTDescr, sift_idx,
		segs_rgb, segs_normals, segs_shot_rgb, segs_shot_normals, segs_SHOTDescr, segs_sift_rgb, segs_sift_normals, segs_SIFTDescr);
	
	recog_result.resize(segs_rgb.size());
	for( int i = 0 ; i < recog_result.size() ; i++ )
		recog_result[i] = Eigen::Matrix4f::Identity();
}

void SegRecog::LoadScene(const std::vector<pcl::PointCloud<PointT>::Ptr, Eigen::aligned_allocator<pcl::PointCloud<PointT>::Ptr>> &segs_rgb_,
				const std::vector<pcl::PointCloud<NormalT>::Ptr, Eigen::aligned_allocator<pcl::PointCloud<NormalT>::Ptr>> &segs_normals_,
				const std::vector<pcl::PointCloud<PointT>::Ptr, Eigen::aligned_allocator<pcl::PointCloud<PointT>::Ptr>> &segs_shot_rgb_,
				const std::vector<pcl::PointCloud<NormalT>::Ptr, Eigen::aligned_allocator<pcl::PointCloud<NormalT>::Ptr>> &segs_shot_normals_,
				const std::vector<pcl::PointCloud<SHOTdescr>::Ptr, Eigen::aligned_allocator<pcl::PointCloud<SHOTdescr>::Ptr>> &segs_SHOTDescr_,
				const std::vector<pcl::PointCloud<PointT>::Ptr, Eigen::aligned_allocator<pcl::PointCloud<PointT>::Ptr>> &segs_sift_rgb_,
				const std::vector<pcl::PointCloud<NormalT>::Ptr, Eigen::aligned_allocator<pcl::PointCloud<NormalT>::Ptr>> &segs_sift_normals_,
				const std::vector<pcl::PointCloud<SIFTDescr>::Ptr, Eigen::aligned_allocator<pcl::PointCloud<SIFTDescr>::Ptr>> &segs_SIFTDescr_)
{
	segs_rgb = segs_rgb_;
	segs_normals = segs_normals_;
	//segs_shot_rgb = segs_shot_rgb_;
	//segs_shot_normals = segs_shot_normals_;
	//segs_SHOTDescr = segs_SHOTDescr_;
	segs_sift_rgb = segs_sift_rgb_;
	segs_sift_normals = segs_sift_normals_;
	segs_SIFTDescr = segs_SIFTDescr_;

	for( int i = 0 ; i < segs_rgb.size(); i++ )
	{
		///////////////////////////////////////////////////////////////////////////////////////////////////////////////////
		pcl::PointCloud<PointT>::Ptr temp_rgb = segs_rgb[i];
		pcl::PointCloud<NormalT>::Ptr temp_normals = segs_normals[i];
		pcl::PointCloud<PointT>::Ptr temp_shot(new pcl::PointCloud<PointT>());
		pcl::PointCloud<NormalT>::Ptr temp_shot_normals(new pcl::PointCloud<NormalT>());
		pcl::PointCloud<SHOTdescr>::Ptr temp_SHOTDescr(new pcl::PointCloud<SHOTdescr>());
		///////////////////////////////////////////////////////////////////////////////////////////////////////////////////
		pcl::VoxelGrid<PointT> sor;
		sor.setInputCloud(temp_rgb);
		sor.setLeafSize(resolution*part_sample_ratio, resolution*part_sample_ratio, resolution*part_sample_ratio);
		sor.filter(*temp_shot);
		computeKeyNormals(temp_shot, temp_shot_normals, temp_rgb, resolution*normal_ratio);

		pcl::SHOTEstimationOMP<PointT, NormalT, SHOTdescr, pcl::ReferenceFrame> descr_est;
		descr_est.setRadiusSearch(resolution*CSHOT_ratio);
		descr_est.setNumberOfThreads(THREADNUM);
		descr_est.setSearchSurface(temp_rgb);
		descr_est.setInputNormals(temp_normals);
		descr_est.setInputCloud(temp_shot);
		descr_est.compute(*temp_SHOTDescr);

		segs_shot_rgb.push_back(temp_shot);
		segs_shot_normals.push_back(temp_shot_normals);
		segs_SHOTDescr.push_back(temp_SHOTDescr);
	}

	recog_result.resize(segs_rgb.size());
	for( int i = 0 ; i < recog_result.size() ; i++ )
		recog_result[i] = Eigen::Matrix4f::Identity();
}

std::vector <pcl::PointIndices> SegRecog::sceneSegmentation(pcl::PointCloud<PointT>::Ptr scene_rgb, pcl::PointCloud<NormalT>::Ptr scene_normals)
{
	clock_t t1, t2;
	t1 = clock();
	pcl::search::KdTree<PointT>::Ptr tree(new pcl::search::KdTree<PointT>());
	
	pcl::RegionGrowing<PointT, NormalT> reg;
	reg.setMinClusterSize(1);
	reg.setMaxClusterSize(10000000);
	reg.setNumberOfNeighbours(searchNeighs);
	reg.setSearchMethod(tree);
	reg.setInputCloud(scene_rgb);
	reg.setInputNormals(scene_normals);
	reg.setSmoothnessThreshold (normalDiff / 180.0 * M_PI);
	reg.setCurvatureTestFlag(false);
	reg.setCurvatureThreshold (1.0);
	
	std::vector <pcl::PointIndices> pre_clusters;
	reg.extract (pre_clusters);

	pcl::ExtractIndices<PointT> extract_rgb;
	extract_rgb.setInputCloud(scene_rgb);
	extract_rgb.setNegative(false);
	std::vector<pcl::PointCloud<PointT>::Ptr, Eigen::aligned_allocator<pcl::PointCloud<PointT>::Ptr>> main_segs_rgb;
	std::vector<pcl::PointCloud<PointT>::Ptr, Eigen::aligned_allocator<pcl::PointCloud<PointT>::Ptr>> sub_segs_rgb;
	std::vector<PointT, Eigen::aligned_allocator<PointT>> sub_center;

	pcl::PointCloud<PointT>::Ptr main_cloud(new pcl::PointCloud<PointT>());
	std::vector<int> labels;
		
	std::vector <pcl::PointIndices> main_clusters;
	std::vector <pcl::PointIndices> sub_clusters;
	for( int j = 0, label_count = 0 ; j < pre_clusters.size() ; j++ )
	{
		pcl::PointIndices::Ptr clusterPt(new pcl::PointIndices());
		clusterPt->indices = pre_clusters[j].indices;
		clusterPt->header = pre_clusters[j].header;
		extract_rgb.setIndices(clusterPt);
		pcl::PointCloud<PointT>::Ptr temp_rgb(new pcl::PointCloud<PointT>());
		extract_rgb.filter(*temp_rgb);

		if( temp_rgb->size() >= minSegment )
		{
			main_segs_rgb.push_back(temp_rgb);
			main_cloud->insert(main_cloud->end(), temp_rgb->begin(), temp_rgb->end());
			labels.insert(labels.end(), temp_rgb->size(), label_count);
			main_clusters.push_back(*clusterPt);
			label_count++;
		}
		else
		{
			sub_segs_rgb.push_back(temp_rgb);
			sub_clusters.push_back(*clusterPt);
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
				exit(0);
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
			main_clusters.at(merged_idx).indices.insert(main_clusters.at(merged_idx).indices.end(), sub_clusters.at(j).indices.begin(), sub_clusters.at(j).indices.end());
			main_segs_rgb.at(merged_idx)->insert(main_segs_rgb.at(merged_idx)->end(), sub_segs_rgb.at(j)->begin(), sub_segs_rgb.at(j)->end());
		}
	}
	std::vector <pcl::PointIndices> final_clusters;
	for( int j = 0; j < main_clusters.size() ; j++ )
		if( main_clusters[j].indices.size() <= maxSegment)
			final_clusters.push_back(main_clusters[j]);

	t2 = clock();
	std::cerr<<"Segmentation Time Elapsed: "<<((double)t2-(double)t1)/CLOCKS_PER_SEC<<std::endl;
	
	std::cerr<<"Cluster Number: "<<final_clusters.size()<<std::endl;
	std::cerr<<std::endl;
	
#ifdef SEGSHOW
	std::srand(time(NULL));
	for( int j = 0 ; j < main_segs_rgb.size() ; j++ )
	{
		std::ostringstream convert1;     // stream used for the conversion
		convert1 << j;

		double r = rand()%255 / 255.0;
		double g = rand()%255 / 255.0;
		double b = rand()%255 / 255.0;
		viewer->addPointCloud(main_segs_rgb[j], "seg_result"+convert1.str());
		
		viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_COLOR, r,g,b, "seg_result"+convert1.str());
	}
	viewer->spin();
#endif
	return final_clusters;

}

void SegRecog::extractSegments(pcl::PointCloud<PointT>::Ptr scene_rgb, pcl::PointCloud<NormalT>::Ptr scene_normals, 
		pcl::PointCloud<PointT>::Ptr scene_sift, pcl::PointCloud<NormalT>::Ptr scene_sift_normals, pcl::PointCloud<SIFTDescr>::Ptr scene_SIFTDescr, std::vector<int> sift_idx,
		std::vector<pcl::PointCloud<PointT>::Ptr, Eigen::aligned_allocator<pcl::PointCloud<PointT>::Ptr>> &segs_rgb,
		std::vector<pcl::PointCloud<NormalT>::Ptr, Eigen::aligned_allocator<pcl::PointCloud<NormalT>::Ptr>> &segs_normals,
		std::vector<pcl::PointCloud<PointT>::Ptr, Eigen::aligned_allocator<pcl::PointCloud<PointT>::Ptr>> &segs_shot_rgb,
		std::vector<pcl::PointCloud<NormalT>::Ptr, Eigen::aligned_allocator<pcl::PointCloud<NormalT>::Ptr>> &segs_shot_normals,
		std::vector<pcl::PointCloud<SHOTdescr>::Ptr, Eigen::aligned_allocator<pcl::PointCloud<SHOTdescr>::Ptr>> &segs_SHOTDescr,
		std::vector<pcl::PointCloud<PointT>::Ptr, Eigen::aligned_allocator<pcl::PointCloud<PointT>::Ptr>> &segs_sift_rgb,
		std::vector<pcl::PointCloud<NormalT>::Ptr, Eigen::aligned_allocator<pcl::PointCloud<NormalT>::Ptr>> &segs_sift_normals,
		std::vector<pcl::PointCloud<SIFTDescr>::Ptr, Eigen::aligned_allocator<pcl::PointCloud<SIFTDescr>::Ptr>> &segs_SIFTDescr)
{

	std::vector <pcl::PointIndices> clusters = sceneSegmentation(scene_rgb, scene_normals);
	
	pcl::ExtractIndices<PointT> extract_rgb;
	pcl::ExtractIndices<NormalT> extract_normals;
		
	extract_rgb.setInputCloud(scene_rgb);
	extract_normals.setInputCloud(scene_normals);

	extract_rgb.setNegative(false);
	extract_normals.setNegative(false);
	
	for( int i = 0 ; i < clusters.size(); i++ )
	{
		pcl::PointIndices::Ptr clusterPt(new pcl::PointIndices());
		clusterPt->indices = clusters[i].indices;
		clusterPt->header = clusters[i].header;

		///////////////////////////////////////////////////////////////////////////////////////////////////////////////////
		pcl::PointCloud<PointT>::Ptr temp_rgb(new pcl::PointCloud<PointT>());
		pcl::PointCloud<NormalT>::Ptr temp_normals(new pcl::PointCloud<NormalT>());
		pcl::PointCloud<PointT>::Ptr temp_shot(new pcl::PointCloud<PointT>());
		pcl::PointCloud<NormalT>::Ptr temp_shot_normals(new pcl::PointCloud<NormalT>());
		pcl::PointCloud<SHOTdescr>::Ptr temp_SHOTDescr(new pcl::PointCloud<SHOTdescr>());
		pcl::PointCloud<PointT>::Ptr temp_sift_rgb(new pcl::PointCloud<PointT>());
		pcl::PointCloud<NormalT>::Ptr temp_sift_normals(new pcl::PointCloud<NormalT>());
		pcl::PointCloud<SIFTDescr>::Ptr temp_sift_descr(new pcl::PointCloud<SIFTDescr>());
		//////////////////////////////////////////////////////////////////////////////////////////////////////////////////
		extract_rgb.setIndices(clusterPt);
		extract_rgb.filter(*temp_rgb);
		segs_rgb.push_back(temp_rgb);

		extract_normals.setIndices(clusterPt);
		extract_normals.filter(*temp_normals);
		segs_normals.push_back(temp_normals);
		///////////////////////////////////////////////////////////////////////////////////////////////////////////////////
		
		///////////////////////////////////////////////////////////////////////////////////////////////////////////////////
		pcl::VoxelGrid<PointT> sor;
		sor.setInputCloud(temp_rgb);
		sor.setLeafSize(resolution*part_sample_ratio, resolution*part_sample_ratio, resolution*part_sample_ratio);
		sor.filter(*temp_shot);
		computeKeyNormals(temp_shot, temp_shot_normals, temp_rgb, resolution*normal_ratio);

		pcl::SHOTEstimationOMP<PointT, NormalT, SHOTdescr, pcl::ReferenceFrame> descr_est;
		descr_est.setRadiusSearch(resolution*CSHOT_ratio);
		descr_est.setNumberOfThreads(THREADNUM);
		descr_est.setSearchSurface(temp_rgb);
		descr_est.setInputNormals(temp_normals);
		descr_est.setInputCloud(temp_shot);
		descr_est.compute(*temp_SHOTDescr);

		segs_shot_rgb.push_back(temp_shot);
		segs_shot_normals.push_back(temp_shot_normals);
		segs_SHOTDescr.push_back(temp_SHOTDescr);
		///////////////////////////////////////////////////////////////////////////////////////////////////////////////////
		
		// Initialize sift tree
		pcl::search::KdTree<PointT>::Ptr sift_tree(new pcl::search::KdTree<PointT>());
		sift_tree->setInputCloud(scene_sift);
		std::vector<int> ind = clusterPt->indices;
		std::vector<int> neigh_indices (1);
		std::vector<float> neigh_sqr_dists (1);
		
		for( int j = 0 ; j < ind.size(); j++ )
		{
			if( sift_tree->nearestKSearch(scene_rgb->points[ind[j]], 1, neigh_indices, neigh_sqr_dists) > 0 && neigh_sqr_dists[0] == 0 )
			{
				temp_sift_rgb->points.push_back(scene_sift->points[neigh_indices[0]]);
				temp_sift_normals->points.push_back(scene_sift_normals->points[neigh_indices[0]]);
				temp_sift_descr->points.push_back(scene_SIFTDescr->points[neigh_indices[0]]);
			}
		}

		segs_sift_rgb.push_back(temp_sift_rgb);
		segs_sift_normals.push_back(temp_sift_normals);
		segs_SIFTDescr.push_back(temp_sift_descr);
		///////////////////////////////////////////////////////////////////////////////////////////////////////////////////
		
	}
}


pcl::CorrespondencesPtr SegRecog::testGuess(pcl::PointCloud<pcl::PointXYZHSV>::Ptr part_hue_, pcl::PointCloud<pcl::PointXYZHSV>::Ptr model_hue_, 
		pcl::PointCloud<NormalT>::Ptr part_normals_, pcl::PointCloud<NormalT>::Ptr model_normals_, float hueT, float curveT, float distT)	//resolution distT
{
	pcl::search::KdTree<pcl::PointXYZHSV> tree;
	tree.setInputCloud(model_hue_);

	int K = 5;
	//float curveT = 0.03;
	//float hueT = 0.08;
	float radius = resolution*K;
	float inlierT = distT*distT;
	
	int partNum = part_hue_->points.size();
	pcl::CorrespondencesPtr corrsPool[8];
	for ( int i = 0 ; i < 8; i++ ){
		pcl::CorrespondencesPtr temp_corrs(new pcl::Correspondences ());
		corrsPool[i] = temp_corrs;
	}
	//omp_set_num_threads(1);
	#pragma omp parallel firstprivate(tree, radius, distT, part_hue_, model_hue_, part_normals_, model_normals_, corrsPool, curveT, hueT) 
	{	
		pcl::PointCloud<NormalT>::Ptr model_normals(new pcl::PointCloud<NormalT>());
		pcl::PointCloud<pcl::PointXYZHSV>::Ptr model_hue(new pcl::PointCloud<pcl::PointXYZHSV>);
		pcl::PointCloud<NormalT>::Ptr part_normals(new pcl::PointCloud<NormalT>());	
		pcl::PointCloud<pcl::PointXYZHSV>::Ptr part_hue(new pcl::PointCloud<pcl::PointXYZHSV>);
		
		pcl::copyPointCloud(*model_normals_, *model_normals);
		pcl::copyPointCloud(*model_hue_, *model_hue);
		pcl::copyPointCloud(*part_normals_, *part_normals);
		pcl::copyPointCloud(*part_hue_, *part_hue);
		
		#pragma omp for
		for( int i = 0 ; i < partNum ; i++ )
		{
			std::vector<int> pointIdx;
			std::vector<float> pointDistance;
			pcl::PointXYZHSV partPt(part_hue->points[i]);
		
			if ( tree.radiusSearch (partPt, radius, pointIdx, pointDistance) > 0 )
			{
				float diffmin = 1000;
				int diffmin_idx = -1;
				
				float part_curvature = part_normals->points[i].curvature;
				for (size_t j = 0; j < pointIdx.size (); j++)
				{
					//std::cerr<<"Finding..."<<scene_hue->points[pointIdx[j]].h <<" "<<model_hue->points[i].h<<std::endl;
					if( pointDistance.at(j) < inlierT )
					{
						pcl::PointXYZHSV modelPt(model_hue->points[pointIdx[j]]);

						float diffh = std::min( fabs(modelPt.h  - partPt.h), 
									std::min(fabs(modelPt.h - 1 - partPt.h), fabs(modelPt.h + 1 - partPt.h)));
						float diffs = fabs(modelPt.s - partPt.s);
						float diffv = fabs(modelPt.v - partPt.v);
						float diffsum = (diffh*2+diffs+diffv)/4;
						float diffcurvature = fabs( model_normals->points[pointIdx[j]].curvature - part_curvature);
						//std::cerr<<diffsum<<std::endl;
						if( diffcurvature < curveT && diffmin > diffsum )
						{
							diffmin = diffsum;
							diffmin_idx = j;
						}
					}	
				}
				//std::cerr<<diffcurvature<<" ";
				//std::cerr<<diffmin<<" ";
				if( diffmin <= hueT )
				{
					pcl::Correspondence temp;
					temp.index_query = i;
					temp.index_match = pointIdx[diffmin_idx];
					temp.distance = pointDistance.at(diffmin_idx);
					//part_model_corrs->push_back(temp);
					corrsPool[omp_get_thread_num()]->push_back(temp);
				}
			}
		}
	}
	pcl::CorrespondencesPtr part_model_corrs(new pcl::Correspondences ());
	for( int i = 0 ; i < 8 ; i++ )
		part_model_corrs->insert(part_model_corrs->begin(), corrsPool[i]->begin(), corrsPool[i]->end());
	return part_model_corrs;
	
}

float SegRecog::myICP(pcl::PointCloud<PointT>::Ptr model_rgb, pcl::PointCloud<PointT>::Ptr scene_rgb, pcl::PointCloud<pcl::PointXYZHSV>::Ptr model_hue, pcl::PointCloud<pcl::PointXYZHSV>::Ptr scene_hue, pcl::PointCloud<NormalT>::Ptr model_normals, pcl::PointCloud<NormalT>::Ptr scene_normals, 
		const Eigen::Matrix4f& initial_guess, Eigen::Matrix4f& rotatranslation) //float model_resolution)
{
	//if the ICP fail to converge, return false. Otherwise, return true!
	int K = 5, Iter_num = 500;
	
	pcl::transformPointCloud(*model_hue, *model_hue, initial_guess);
	
	rotatranslation = initial_guess;
	pcl::search::KdTree<pcl::PointXYZHSV> scene_tree;
	scene_tree.setInputCloud(scene_hue);
	pcl::registration::TransformationEstimationSVD<pcl::PointXYZHSV, pcl::PointXYZHSV> SVD;
	double last_error = 100000;
	for( size_t iter = 0 ; iter < Iter_num ; iter++ )
	{
		//In each round of ICP, transform the model_hue cloud
		
		Eigen::Matrix4f svdRt;
		//clock_t t1, t2;
		//t1 = clock();
		pcl::CorrespondencesPtr model_scene_corrs = testGuess(model_hue, scene_hue, model_normals, scene_normals, hueT, curveT, distT); //
		//t2 = clock();
		//std::cerr<<"Time Elapsed1: "<<((double)t2-(double)t1)/CLOCKS_PER_SEC<<std::endl;
		
		SVD.estimateRigidTransformation(*model_hue, *scene_hue, *model_scene_corrs, svdRt);
		pcl::transformPointCloud(*model_hue, *model_hue, svdRt);
		rotatranslation = svdRt * rotatranslation ;
		std::cerr<<"Ratio "<<(model_scene_corrs->size()+0.0) / model_hue->points.size()<<std::endl;
		
		if( (model_scene_corrs->size()+0.0) / model_hue->points.size() >= corrs_ratio ) //sufficient inlier found
		{
			size_t corrs = model_scene_corrs->size();
			double this_error=0;
			
			for( size_t j = 0; j < corrs; j++ )
			{
				pcl::PointXYZHSV model_pt = model_hue->points[model_scene_corrs->at(j).index_query];
				pcl::PointXYZHSV scene_pt = scene_hue->points[model_scene_corrs->at(j).index_match];
				NormalT normal_temp = scene_normals->points[model_scene_corrs->at(j).index_match];
				double diffx = model_pt.x - scene_pt.x, diffy = model_pt.y - scene_pt.y, diffz = model_pt.z - scene_pt.z;
				//double diffx = (model_pt.x - scene_pt.x)*normal_temp.normal_x, 
				//	   diffy = (model_pt.y - scene_pt.y)*normal_temp.normal_y, 
				//	   diffz = (model_pt.z - scene_pt.z)*normal_temp.normal_z;
				double dist = sqrt( diffx*diffx + diffy*diffy + diffz*diffz );

				this_error += dist;
			}
			this_error = this_error / corrs;
			//std::cerr<<"This Error: "<<fabs(this_error - last_error)<<" "<<this_error<<std::endl;
			if( fabs(this_error - last_error) < 0.000001 )  //Convergence reach
			{
				std::cerr<<"Convergence Reached. Error: "<<this_error<<std::endl;
				std::cerr<<"Iter Num: "<<iter<<std::endl;
				return (model_scene_corrs->size()+0.0) / model_hue->points.size();
				/*
				int total_bins = h_bin*s_bin*i_bin;

				float *model_hist = new float[total_bins];
				float *scene_hist = new float[total_bins];
				memset(model_hist, 0, sizeof(float)*total_bins);
				memset(scene_hist, 0, sizeof(float)*total_bins);
				int model_idx, scene_idx;
				for(int j = 0 ; j < corrs; j++ )
				{
					pcl::PointXYZHSV model_pt = model_hue->points[model_scene_corrs->at(j).index_query];
					pcl::PointXYZHSV scene_pt = scene_hue->points[model_scene_corrs->at(j).index_match];
					//PointT model_pt = model_rgb->points[model_scene_corrs->at(j).index_query];
					//PointT scene_pt = scene_rgb->points[model_scene_corrs->at(j).index_match];
					model_idx = (int)(model_pt.h*h_bin) + (int)(model_pt.s*s_bin)*h_bin + (int)(model_pt.v*i_bin)*h_bin*s_bin;
					scene_idx = (int)(scene_pt.h*h_bin) + (int)(scene_pt.s*s_bin)*h_bin + (int)(scene_pt.v*i_bin)*h_bin*s_bin;
					model_hist[model_idx < total_bins? model_idx:total_bins] += 1;
					scene_hist[scene_idx < total_bins? scene_idx:total_bins] += 1;
					//model_hist[(int)(model_pt.r/255.0*h_bin) + (int)(model_pt.g/255.0*s_bin)*h_bin + (int)(model_pt.b/255.0*v_bin)*h_bin*s_bin] += 1;
					//scene_hist[(int)(scene_pt.r/255.0*h_bin) + (int)(scene_pt.g/255.0*s_bin)*h_bin + (int)(scene_pt.b/255.0*v_bin)*h_bin*s_bin] += 1;
				}
				
				float sim_score = 0;
				for( int j = 0; j < total_bins; j++ )
					sim_score += sqrt(model_hist[j]*scene_hist[j]);
				delete []model_hist;
				delete []scene_hist;
				std::cerr<<"Sim Score: "<<sim_score / corrs<<std::endl;
				if( sim_score / corrs >= histT )
					return (model_scene_corrs->size()+0.0) / model_hue->points.size();
				else
					return 0.0;
					*/
			}
			else
				last_error = this_error;
			
		}
		else
			break;
	}
	
	return 0.0;
}

float SegRecog::distPt(const PointT &pt1, const PointT &pt2)
{
	float diffx = pt1.x - pt2.x;
	float diffy = pt1.y - pt2.y;
	float diffz = pt1.z - pt2.z;
	return sqrt(diffx*diffx + diffy*diffy + diffz*diffz);
}

bool SegRecog::testTransPt(pcl::PointCloud<PointT>::Ptr src, pcl::PointCloud<PointT>::Ptr dst, float inlierT)
{
	if( src->size() != dst->size() )
		return false;
	bool flag = true;
	int num = src->size();
	for( int i = 0 ; i < num; i++ ){
		if( distPt(src->points[i], dst->points[i]) > inlierT )
			return false;
	}
	return true;
}

void SegRecog::GenNormalPt(const pcl::PointCloud<PointT>::Ptr cloud, const pcl::PointCloud<NormalT>::Ptr cloud_normals, std::vector<int> idx, pcl::PointCloud<PointT>::Ptr candidates)
{
	for( int i = 0 ; i < idx.size(); i++ )
	{
		PointT pt = cloud->points[idx[i]];
		NormalT pt_normal = cloud_normals->points[idx[i]];

		PointT temp = pt;
		if( pt_normal.normal_x != pt_normal.normal_x )
		{
			temp.x = pt.x;
			temp.y = pt.y;
			temp.z = pt.z;
		}
		else
		{
			temp.x = pt.x + pt_normal.normal_x*0.1;
			temp.y = pt.y + pt_normal.normal_y*0.1;
			temp.z = pt.z + pt_normal.normal_z*0.1;
		}
		candidates->points.push_back(pt);
		candidates->points.push_back(temp);
	}
}

float SegRecog::testShapeDist(const pcl::PointCloud<PointT>::Ptr cloud, const pcl::search::KdTree<PointT>::Ptr tree)
{
	int num = cloud->size();
	float dist_sum = 0, mean, dist_temp;
	std::vector<float> dist;
	std::vector<int> pointIdx;
	std::vector<float> pointDistance;

	for( int i = 0 ; i < num ; i++ )
	{
		if( tree->nearestKSearch(cloud->points[i], 1, pointIdx, pointDistance) > 0 )
		{
			dist_temp = sqrt(pointDistance[0]);
			dist_sum += dist_temp;
			dist.push_back(dist_temp);
		}
	}
	mean = dist_sum / dist.size();
	float variance=0;
	for( int i= 0 ; i < dist.size() ; i++ )
	{
		float var = dist[i] - mean;
		variance += var*var;
	}
#ifdef DEBUG
	std::cerr<<"Mean: "<<mean<<" Variance: "<<sqrt(variance/dist.size())<<std::endl;
#endif
	return sqrt(variance/dist.size());

}

// SHOT recognition pipeline
std::vector<std::vector<int>> SegRecog::List_SHOT_Corrs(pcl::PointCloud<SHOTdescr>::Ptr part_shot, pcl::PointCloud<SHOTdescr>::Ptr model_shot)
{
	int K = 10, i;
	int partNum = part_shot->size();
	
	std::vector<std::vector<int>> corrsPool(partNum);
	pcl::search::KdTree<SHOTdescr> tree;
	tree.setInputCloud(model_shot);		//build kdtree for larger point cloud, ususally for the full model

	clock_t t1, t2;
	t1 = clock();
	#pragma omp parallel for private(i) firstprivate(K, tree, partNum)  
	for( i = 0 ; i < partNum ; i++ ){
		std::vector<int> neigh_indices (K);
		std::vector<float> neigh_sqr_dists (K);
		
		int found_neighs = tree.nearestKSearch(part_shot->at (i), K, neigh_indices, neigh_sqr_dists);
		/*
		int j;
		for( j = 0 ; j < found_neighs ; j++ )
			if( neigh_sqr_dists[j] > keysT )
				break;
		if( j > 0)
		{
			corrsPool_[i].resize(j-1);
			std::copy(neigh_indices.begin(), neigh_indices.begin()+j-1, corrsPool_[i].begin());
		}
		*/
		corrsPool[i] = neigh_indices;
	}
	t2 = clock();
	std::cerr<<"Building SHOT Corrs Elapsed: "<<((double)t2-(double)t1)/CLOCKS_PER_SEC<<std::endl;
	/*//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	int valid_corrs = 0;
	for( i = 0 ; i < partNum ; i++ )
		if( !corrsPool_[i].empty() )
			valid_corrs++;
	float valid_ratio = valid_corrs/(partNum+0.0);
	std::cerr<<"Valid Correspondences: "<<valid_ratio<<std::endl;
	if( valid_ratio < 0.3 )
		return Eigen::Matrix4f::Identity();
	/*//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	return corrsPool;
}

Eigen::Matrix4f SegRecog::poseSHOTGenerator(pcl::PointCloud<PointT>::Ptr part_, pcl::PointCloud<PointT>::Ptr model_, pcl::PointCloud<NormalT>::Ptr part_normals_, pcl::PointCloud<NormalT>::Ptr model_normals_, 
				pcl::PointCloud<PointT>::Ptr dense_model_, const std::vector<std::vector<int>> &corrsPool_)
{
	int partNum = part_->size(), modelNum = model_->size();
	int i, iterMax = 50000, subIter = 50;
	float minSampleDist = 0.01, inlierT = 0.01, scoreT = 0.01;

	//RANSAC Process
	Eigen::Matrix4f trans[THREADNUM];
	Eigen::Matrix4f *transP = trans;
	float score[THREADNUM], MAX_SCORE = 100;
	for( int i = 0 ; i < THREADNUM; i++ )
		score[i] = MAX_SCORE;		//set maximum limit
	float *scoreP = score;

#ifdef DEBUG
	omp_set_num_threads(1);
#else
	omp_set_num_threads(THREADNUM);
#endif
	#pragma omp parallel firstprivate(partNum, part_, part_normals_, model_, model_normals_, dense_model_, inlierT, minSampleDist, subIter, scoreT, transP, scoreP, iterMax)
	{
		int thread_idx = omp_get_thread_num()+1, i;
		srand(time(NULL)*thread_idx*thread_idx*thread_idx);		//reset random seed for each thread

		pcl::PointCloud<PointT>::Ptr model(new pcl::PointCloud<PointT>());
		pcl::PointCloud<PointT>::Ptr part(new pcl::PointCloud<PointT>());	
		pcl::PointCloud<PointT>::Ptr dense_model(new pcl::PointCloud<PointT>());
		pcl::PointCloud<NormalT>::Ptr model_normals(new pcl::PointCloud<NormalT>());
		pcl::PointCloud<NormalT>::Ptr part_normals(new pcl::PointCloud<NormalT>());	
		pcl::search::KdTree<PointT>::Ptr dense_tree(new pcl::search::KdTree<PointT>());
		std::vector<std::vector<int>> corrsPool(corrsPool_.size());

		#pragma omp critical
		{
			pcl::copyPointCloud(*model_, *model);
			pcl::copyPointCloud(*part_, *part);
			pcl::copyPointCloud(*dense_model_, *dense_model);
			pcl::copyPointCloud(*model_normals_, *model_normals);
			pcl::copyPointCloud(*part_normals_, *part_normals);
			std::copy(corrsPool_.begin(), corrsPool_.end(), corrsPool.begin());
		}
		dense_tree->setInputCloud(dense_model);

		pcl::CorrespondencesPtr corr (new pcl::Correspondences ());
		for(i = 0 ; i < 6 ; i++ ){
			pcl::Correspondence temp(i,i,0);
			corr->push_back(temp);
		}
		pcl::registration::TransformationEstimationSVD<PointT, PointT> SVD;
		//std::cerr<<omp_get_thread_num()<<std::endl;
		#pragma omp for 
		for( i = 0; i < iterMax; i++)
		{
			//Local Variables Initialization
			int idx1, idx2, idx3;
			std::vector<int> src_idx;			
			
			PointT pt1, pt2, pt3;
			
			idx1 = std::rand()%partNum;
			src_idx.push_back(idx1);
			pt1 = part->points[idx1];

			int count = 0;
			while(true)
			{
				count++;
				if(count > 1000)
					std::cerr<<"Fuck!"<<std::endl;
				idx2 = std::rand()%partNum;
				if( distPt(pt1, part->points[idx2]) > minSampleDist )
				{
					pt2 = part->points[idx2];
					src_idx.push_back(idx2);
					break;
				}
			}
			count = 0;
			while(true)
			{
				count++;
				if(count > 1000)
					std::cerr<<"Fuck!"<<std::endl;
				idx3 = std::rand()%partNum;
				pt3 = part->points[idx3];
				if( distPt(pt1, pt3) > minSampleDist && distPt(pt2, pt3) > minSampleDist )
				{
					src_idx.push_back(idx3);
					break;
				}
			}
			pcl::PointCloud<PointT>::Ptr src(new pcl::PointCloud<PointT>());
			GenNormalPt(part, part_normals, src_idx, src);
			
			for( int j = 0 ; j < subIter ; j++ )
			{
				pcl::PointCloud<PointT>::Ptr dst(new pcl::PointCloud<PointT>());
				std::vector<int> dst_idx(3);
				if( j == 0)
				{
					dst_idx[0] = corrsPool[idx1].at(0);
					dst_idx[1] = corrsPool[idx2].at(0);
					dst_idx[2] = corrsPool[idx3].at(0);
				}
				else
				{
					dst_idx[0] = corrsPool[idx1].at(std::rand()%corrsPool[idx1].size());
					dst_idx[1] = corrsPool[idx2].at(std::rand()%corrsPool[idx2].size());
					dst_idx[2] = corrsPool[idx3].at(std::rand()%corrsPool[idx3].size());
				}
				GenNormalPt(model, model_normals, dst_idx, dst);
				
				Eigen::Matrix4f trans;
				SVD.estimateRigidTransformation(*src, *dst, *corr, trans);

				pcl::PointCloud<PointT>::Ptr test(new pcl::PointCloud<PointT>());
				pcl::transformPointCloud(*src, *test, trans);
				if( testTransPt(test,dst,inlierT) )
				{
					pcl::PointCloud<PointT>::Ptr temp_part(new pcl::PointCloud<PointT>);
					pcl::transformPointCloud(*part, *temp_part, trans);
					float shapeDist = testShapeDist(temp_part, dense_tree);		
					//pcl::CorrespondencesPtr part_dense_corrs = testGuess(temp_hue, dense_hue, part_normals, dense_normals, resolution, 0.5);
					//float score = (part_dense_corrs->size()+0.0) / part_hue->size();
					
#ifdef DEBUG
					//std::cout<<shapeDist<<" "<<i<<" "<<j<<" "<<omp_get_thread_num()<<std::endl;
					pcl::PointCloud<PointT>::Ptr result(new pcl::PointCloud<PointT>());
					pcl::transformPointCloud(*part, *result, trans); 
					viewer->removePointCloud("result");
					viewer->addPointCloud(result, "result");
					viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_COLOR, 255, 255, 255, "result");

					viewer->removePointCloud("part_keys");
					viewer->addPointCloud(test, "part_keys");
					viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_COLOR, 255, 0, 0, "part_keys");
					viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 5, "part_keys");

					viewer->removePointCloud("model_keys");
					viewer->addPointCloud(dst, "model_keys");
					viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_COLOR, 0, 0, 255, "model_keys");
					viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 5, "model_keys");
					viewer->spin();
#endif		
					if( shapeDist <= scoreT )
					{
						if( shapeDist < scoreP[omp_get_thread_num()])
						{
							scoreP[omp_get_thread_num()] = shapeDist;
							transP[omp_get_thread_num()] = trans;
						}
						//break;
					}
				
				}
			}
		}
	}
	float bestS = MAX_SCORE;
	Eigen::Matrix4f bestT = Eigen::Matrix4f::Identity();
	
	for( int i = 0 ; i < THREADNUM ; i++ )
	{ 
		std::cerr<<score[i]<<std::endl;
		if( score[i] < bestS )
		{
			bestS = score[i];
			bestT = trans[i];
		}
	}
	std::cerr<<"Best Score: "<<bestS<<std::endl;
	return bestT;
}

Eigen::Matrix4f SegRecog::recogSHOT(pcl::PointCloud<PointT>::Ptr part, pcl::PointCloud<PointT>::Ptr model, pcl::PointCloud<NormalT>::Ptr part_normals, pcl::PointCloud<NormalT>::Ptr model_normals,
	pcl::PointCloud<SHOTdescr>::Ptr part_shot, pcl::PointCloud<SHOTdescr>::Ptr model_shot, pcl::PointCloud<PointT>::Ptr dense_model, pcl::PointCloud<NormalT>::Ptr dense_normals)
{
	std::vector<std::vector<int>> corrsPool = List_SHOT_Corrs(part_shot, model_shot);
	Eigen::Matrix4f bestT = poseSHOTGenerator(part, model, part_normals, model_normals, dense_model, corrsPool);

	if( bestT != Eigen::Matrix4f::Identity())
	{
		pcl::PointCloud<pcl::PointXYZHSV>::Ptr part_hue(new pcl::PointCloud<pcl::PointXYZHSV>);
		pcl::PointCloud<pcl::PointXYZHSV>::Ptr dense_hue(new pcl::PointCloud<pcl::PointXYZHSV>);
		ExtractHue(part, part_hue);
		ExtractHue(dense_model, dense_hue);
	
		Eigen::Matrix4f rotatranslation;
		float icpScore = myICP(part, dense_model, part_hue, dense_hue, part_normals, dense_normals, bestT, rotatranslation);
		std::cerr<<"icpScore: "<<icpScore<<std::endl;
		if( icpScore > corrs_ratio )
			return rotatranslation;
	}

	return Eigen::Matrix4f::Identity();
}


//SIFT recognition pipeline
std::vector<std::vector<int>> SegRecog::List_SIFT_Corrs(pcl::PointCloud<SIFTDescr>::Ptr part_SIFTDescr, pcl::PointCloud<SIFTDescr>::Ptr model_SIFTDescr)
{
	int i;
	float sift_ratio = 0.7;
	int partNum = part_SIFTDescr->size();
	int modelNum = model_SIFTDescr->size();
	std::vector<std::vector<int>> corrsPool(partNum);
	
	//#pragma omp parallel for private(i) firstprivate(modelNum, partNum, sift_ratio)  
	for( i = 0 ; i < modelNum ; i++ ){
		float best_score = 100000;
		float second_best_score = best_score+1;
		float match_score;
		int best_idx;
		for( int j=0 ; j<partNum ; j++ )
		{
			//match_score = acos( scene_descr.row(i).dot(model_descr.row(j)));
			float dotProduct = 0;
			for( int k = 0 ; k < 128; k++ )
				dotProduct += model_SIFTDescr->points[i].siftDescr[k] * part_SIFTDescr->points[j].siftDescr[k];
			match_score = acos( dotProduct );

			if (match_score < best_score)
			{
				best_idx = j;
				second_best_score = best_score;
				best_score = match_score;
			}
			else if (match_score < second_best_score)
				second_best_score = match_score;
		}
		
		if( best_score < sift_ratio*second_best_score )
			corrsPool[best_idx].push_back(i);
	}
	
	return corrsPool;
}

Eigen::Matrix4f SegRecog::poseSIFTGenerator(pcl::PointCloud<PointT>::Ptr in_part_sift, pcl::PointCloud<PointT>::Ptr model_sift, pcl::PointCloud<NormalT>::Ptr in_part_sift_normals, pcl::PointCloud<NormalT>::Ptr model_sift_normals, 
				pcl::PointCloud<PointT>::Ptr part, pcl::PointCloud<PointT>::Ptr dense_model, const std::vector<std::vector<int>> &corrsPool)
{
	int partNum = in_part_sift->size(), modelNum = model_sift->size();
	int i, iterMax = SIFT_iter_ratio * partNum * (partNum-1) * (partNum-2);
	iterMax = iterMax <= 500 ? iterMax : 500;
	//iterMax = 500;
	float inlierT = 0.01, scoreMin = 0.01;

	Eigen::Matrix4f bestT = Eigen::Matrix4f::Identity();

	pcl::search::KdTree<PointT>::Ptr dense_tree(new pcl::search::KdTree<PointT>());
	dense_tree->setInputCloud(dense_model);

	std::srand(time(NULL));

	pcl::CorrespondencesPtr corr (new pcl::Correspondences ());
	for(i = 0 ; i < 6 ; i++ ){
		pcl::Correspondence temp(i,i,0);
		corr->push_back(temp);
	}
	pcl::registration::TransformationEstimationSVD<PointT, PointT> SVD;
	
	for( i = 0; i < iterMax; i++)
	{
		//Local Variables Initialization
		int idx1, idx2, idx3;
		std::vector<int> src_idx(3);			
			
		PointT pt1, pt2, pt3;
			
		idx1 = std::rand()%partNum;
		src_idx[0] = idx1;
		pt1 = in_part_sift->points[idx1];

		while(true)
		{
			idx2 = std::rand()%partNum;
			if( idx2 != idx1 )
			{
				pt2 = in_part_sift->points[idx2];
				src_idx[1] = idx2;
				break;
			}
		}
		
		while(true)
		{
			idx3 = std::rand()%partNum;
			if( idx3 != idx2 && idx3 != idx1 )
			{
				pt3 = in_part_sift->points[idx3];
				src_idx[2] = idx3;
				break;
			}
		}
		pcl::PointCloud<PointT>::Ptr src(new pcl::PointCloud<PointT>());
		GenNormalPt(in_part_sift, in_part_sift_normals, src_idx, src);

		int len1 = corrsPool[idx1].size();
		int len2 = corrsPool[idx2].size();
		int len3 = corrsPool[idx3].size();
	
		int subIter = len1 * len2 * len3;
		subIter = subIter < 30 ? subIter : 30; 
		
		for( int j = 0 ; j < subIter ; j++ )
		{
			pcl::PointCloud<PointT>::Ptr dst(new pcl::PointCloud<PointT>());
			std::vector<int> dst_idx(3);
			dst_idx[0] = corrsPool[idx1].at(std::rand()%len1);
			dst_idx[1] = corrsPool[idx2].at(std::rand()%len2);
			dst_idx[2] = corrsPool[idx3].at(std::rand()%len3);
			
			GenNormalPt(model_sift, model_sift_normals, dst_idx, dst);
			
			Eigen::Matrix4f trans;
			SVD.estimateRigidTransformation(*src, *dst, *corr, trans);

			pcl::PointCloud<PointT>::Ptr test(new pcl::PointCloud<PointT>());
			pcl::transformPointCloud(*src, *test, trans);
			if( testTransPt(test,dst,inlierT) )
			{
				pcl::PointCloud<PointT>::Ptr temp_part(new pcl::PointCloud<PointT>);
				pcl::transformPointCloud(*part, *temp_part, trans);
				float shapeDist = testShapeDist(temp_part, dense_tree);		
				
				if( shapeDist <= scoreMin )
				{
					scoreMin = shapeDist;
					bestT = trans;		
					if( scoreMin <= resolution * 2)
					{
						std::cerr<<"Best Score: "<<scoreMin<<std::endl;
						return bestT;
					}
				}
				
			}
		}
	}

	
	std::cerr<<"Best Score: "<<scoreMin<<std::endl;
	return bestT;
}


Eigen::Matrix4f SegRecog::recogSIFT(pcl::PointCloud<PointT>::Ptr part_sift, pcl::PointCloud<NormalT>::Ptr part_sift_normals, pcl::PointCloud<SIFTDescr>::Ptr part_SIFTDescr, 
						pcl::PointCloud<PointT>::Ptr model_sift, pcl::PointCloud<NormalT>::Ptr model_sift_normals, pcl::PointCloud<SIFTDescr>::Ptr model_SIFTDescr, 
						pcl::PointCloud<PointT>::Ptr part, pcl::PointCloud<NormalT>::Ptr part_normals, 
						pcl::PointCloud<PointT>::Ptr dense_model, pcl::PointCloud<NormalT>::Ptr dense_normals)
{
	Eigen::Matrix4f bestT = Eigen::Matrix4f::Identity();
	std::vector<std::vector<int>> corrsPool = List_SIFT_Corrs(part_SIFTDescr, model_SIFTDescr);
	
	pcl::PointCloud<PointT>::Ptr in_part_sift(new pcl::PointCloud<PointT>());
	pcl::PointCloud<NormalT>::Ptr in_part_sift_normals(new pcl::PointCloud<NormalT>());
	std::vector<std::vector<int>> in_corrsPool;

	for( int i=0 ; i < part_SIFTDescr->size(); i++ ){
		if(!corrsPool[i].empty())
		{
			in_corrsPool.push_back(corrsPool[i]);
			in_part_sift->points.push_back(part_sift->points[i]);
			in_part_sift_normals->points.push_back(part_sift_normals->points[i]);
		}
	}
	std::cerr<<"Number of inlier SIFTs: "<<in_corrsPool.size()<<std::endl;
	if( in_corrsPool.size() < 3 )
		return bestT;

	bestT = poseSIFTGenerator(in_part_sift, model_sift, in_part_sift_normals, model_sift_normals, part, dense_model, in_corrsPool);
	
	if( bestT != Eigen::Matrix4f::Identity())
	{
		pcl::PointCloud<pcl::PointXYZHSV>::Ptr part_hue(new pcl::PointCloud<pcl::PointXYZHSV>);
		pcl::PointCloud<pcl::PointXYZHSV>::Ptr dense_hue(new pcl::PointCloud<pcl::PointXYZHSV>);
		ExtractHue(part, part_hue);
		ExtractHue(dense_model, dense_hue);
	
		Eigen::Matrix4f rotatranslation;
		float icpScore = myICP(part, dense_model, part_hue, dense_hue, part_normals, dense_normals, bestT, rotatranslation);
		std::cerr<<"icpScore: "<<icpScore<<std::endl;
		if( icpScore > 0.5 )
			return rotatranslation;
	}

	return Eigen::Matrix4f::Identity();
}

void SegRecog::Recognize()
{
	clock_t t1, t2;
	for( int k = 0 ; k < segs_rgb.size(); k++ )
	{
		std::cerr<<std::endl<<"PART "<<k<<" "<<segs_rgb[k]->size()<<" "<<segs_shot_rgb[k]->size()<<std::endl;

		std::ostringstream convert;
		convert << k;

		pcl::PointCloud<PointT>::Ptr part_rgb = segs_rgb[k];
		pcl::PointCloud<NormalT>::Ptr part_normals = segs_normals[k];
		pcl::PointCloud<PointT>::Ptr part_shot = segs_shot_rgb[k];
		pcl::PointCloud<NormalT>::Ptr part_shot_normals = segs_shot_normals[k];
		pcl::PointCloud<SHOTdescr>::Ptr part_SHOTDescr = segs_SHOTDescr[k];
		pcl::PointCloud<PointT>::Ptr part_sift = segs_sift_rgb[k];
		pcl::PointCloud<NormalT>::Ptr part_sift_normals = segs_sift_normals[k];
		pcl::PointCloud<SIFTDescr>::Ptr part_SIFTDescr = segs_SIFTDescr[k];

#ifdef SEGSHOW
		viewer->removeAllPointClouds();
		viewer->initCameraParameters();
		viewer->addPointCloud(part_shot, "part");
		//viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_COLOR, 255, 255, 255, "part");
		viewer->addPointCloudNormals<PointT, NormalT>(part_sift, part_sift_normals, 1, 0.1, "part_sift");
		viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 4, "part_sift");
		viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_COLOR, 255, 0, 0, "part_sift");
		viewer->spin();
		//continue;
#endif
		if( (part_shot->size()+0.0) / model_shot->size() < 0.18 || (part_shot->size()+0.0) / model_shot->size() > 1.1)
			continue;

		Eigen::Matrix4f finalGuess = Eigen::Matrix4f::Identity();
		if( sift_engine )
		{
			if( part_sift->size() >= 3 ){
				std::cerr<<"Start SIFT RANSACing..."<<std::endl;
				t1 = clock();
				finalGuess = recogSIFT(part_sift, part_sift_normals, part_SIFTDescr, model_sift, model_sift_normals, model_SIFTDescr, 
													part_shot, part_shot_normals, model_rgb, model_normals);			//using downsample part_shot to reduce processing time
				t2 = clock();
				std::cerr<<"SIFT RANSACing Time Elapsed: "<<((double)t2-(double)t1)/CLOCKS_PER_SEC<<std::endl;
			}
		}
		if( finalGuess == Eigen::Matrix4f::Identity() && shot_engine )
		{
			std::cerr<<"Start SHOT RANSACing..."<<std::endl;
			t1 = clock();
			finalGuess = recogSHOT(part_shot, model_shot, part_shot_normals, model_shot_normals, part_SHOTDescr, model_SHOTDescr, model_rgb, model_normals);
			t2 = clock();
			std::cerr<<"SHOT RANSACing Time Elapsed: "<<((double)t2-(double)t1)/CLOCKS_PER_SEC<<std::endl;
		}
		recog_result[k] = finalGuess;
	}
}