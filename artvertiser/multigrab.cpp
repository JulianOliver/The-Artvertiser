#include "multigrab.h"

#include "FProfiler/FProfiler.h"

int W;
int H;

MultiGrab::~MultiGrab()
{
    printf("in ~MultiGrab\n");
    // delete all the cameras;
    for ( int i=0; i<cams.size(); i++ )
    {
        delete cams[i];
    }
    cams.clear();

}

int MultiGrab::init(bool cacheTraining, char *modelfile, char *avi_bg_path, int width, int height, int v4l_device, int detect_width, int detect_height )
{

	CvCapture *c;
	cams.clear();

	// if artvertiser is presented an argv of a background file (-b /path/to/avi), accept it as a capture device.

	if (strlen(avi_bg_path) > 0)
	{
		cout << "MultiGrab::init creating capture from avi " << avi_bg_path << " size " << width << "x" << height << " detect at " << detect_width << "x" << detect_height << endl;
		CvCapture *c = cvCaptureFromAVI(avi_bg_path);
		if ( c == 0 )
		{
		    cerr << "cvCaptureFromAVI return null" << endl;
		    return 0;
		}
		cams.push_back(new Cam(c, width, height, detect_width, detect_height ));
	}
	// else capture from the V4L2 device
	else
	{
		/*
		while (1) {
			CvCapture *c = cvCaptureFromCAM(cams.size());
			if (!c) break;
			cams.push_back(new Cam(c));
		}
		*/
		cout << "MultiGrab::init creating camera capture at " << width << "x" << height << " detect at " << detect_width << "x" << detect_height << endl;
		CvCapture *c = cvCaptureFromCAM(v4l_device/*, width, height*/ );
		cams.push_back(new Cam(c, width, height, detect_width, detect_height ));
	}
	if (cams.size()==0) {
		 return 0;
	}

	if (!model.buildCached(cams.size(), cams[0]->cam, cacheTraining, cams[0]->detector )) {
		cout << "model.buildCached() failed.\n";
		return 0;
	}

	for (int i=1; i<cams.size(); ++i) {
		//! TODO mem to mem copy from cams[0]
		cams[i]->detector.load(string(modelfile)+".bmp.classifier", cams[i]->getKeypointDetectorPtrPtr());
        cams[i]->releaseKeypointDetectorLock();
	}

	W=width;
	H=height;

	return 1;
}

void MultiGrab::grabFrames()
{
    #ifdef USE_MULTITHREADCAPTURE
    #else
	for (vector<Cam *>::iterator it=cams.begin(); it!=cams.end(); ++it)
	{
		cvGrabFrame((*it)->cam);
	}
	#endif
}

void MultiGrab::allocLightCollector()
{
	for (vector<Cam *>::iterator it=cams.begin(); it!=cams.end(); ++it)
		(*it)->lc = new LightCollector(model.map.reflc);
}

void MultiGrab::Cam::setCam(CvCapture *c, int _width, int _height, int _detect_width, int _detect_height ) {
	if (cam)
	{
	    if ( mtc )
            delete mtc;
        mtc = 0;
	    cvReleaseCapture(&cam);
	}
	if (c==0) return;
	cam = c;

	// avoid saturating the firewire bus
	cvSetCaptureProperty(cam, CV_CAP_PROP_FPS, 30);
	cout << "setting cv capture property for size to " << _width << "x" << _height << endl;
	int res1 = cvSetCaptureProperty(cam, CV_CAP_PROP_FRAME_WIDTH, _width);
	int res2 = cvSetCaptureProperty(cam, CV_CAP_PROP_FRAME_HEIGHT, _height);
	printf("cvSetCaptureProperty: results %i %i\n", res1, res2 );

	// optionally downsample the video to accelerate computation
    detect_width = _detect_width;
    detect_height = _detect_height;

    // now mtc
    mtc = new MultiThreadCapture( cam );
    mtc->setupResolution( detect_width, detect_height, /*grayscale*/ 1 );
    mtc->startCapture();
    IplImage* f = NULL;
    int timeout = 5000;
    printf("MultiGrab::Cam::setCam waiting for camera to become ready... (5s timeout)\n");
    bool got = false;
    while ( timeout >= 0 && !got )
    {
        // 50ms jumps
        usleep( 50*1000 );
        timeout -= 50;
        got = mtc->getCopyOfLastFrame( &f );
    }
	assert(f != 0 && "camera did not become ready");

	width = f->width; //cvRound(cvGetCaptureProperty(cam, CV_CAP_PROP_FRAME_WIDTH));
	height = f->height; //cvRound(cvGetCaptureProperty(cam, CV_CAP_PROP_FRAME_HEIGHT));
	double fps = cvGetCaptureProperty(cam, CV_CAP_PROP_FPS);
	cout << "Camera capturing " << width << "x" << height << " at "
		<< fps << " fps, " << f->nChannels << " channels.\n"
		" Detecting at "<< detect_width <<"x" << detect_height << endl;


	cvReleaseImage(&f);

}


MultiGrab::Cam::~Cam() {
	if (gray && gray != frame) cvReleaseImage(&gray);
	if (frame_detectsize && frame_detectsize != gray && frame_detectsize != frame ) cvReleaseImage(&frame_detectsize);
	if (mtc) delete mtc;
	if (cam) cvReleaseCapture(&cam);
	if (lc) delete lc;
}

void MultiGrab::Cam::shutdownMultiThreadCapture()
{
    if ( mtc )
        mtc->stopCapture();
}

// this could be run in a separate thread
bool MultiGrab::Cam::detect()
{
    PROFILE_THIS_FUNCTION();
	CvSize detect_size = cvSize( detect_width, detect_height );

	PROFILE_SECTION_PUSH("frame management");

    PROFILE_SECTION_PUSH("retrieve");
    IplImage *gray_temp, *frame_temp;
    FTime timestamp;
    keypoint* keypoints = NULL;
    int num_keypoints = 0;
    object_view* object_input_view = NULL;
    bool got = mtc->getLastDetectFrame( &gray_temp, &frame_temp, &timestamp,
                                          &keypoints, &num_keypoints, &object_input_view,
                                          /*block until available*/ true );
    PROFILE_SECTION_POP();
    if ( !got || frame_temp == 0 || gray_temp == 0 || object_input_view == 0 )
    {
        PROFILE_SECTION_POP();
        return false;
    }
    gray = gray_temp;
    frame = frame_temp;
    detected_frame_timestamp = timestamp;

    PROFILE_SECTION_POP();


	PROFILE_SECTION_PUSH( "detect" );
	// run the detector
	bool res = false;
	if (detector.detect(gray, keypoints, num_keypoints, object_input_view )) {
	    PROFILE_SECTION_PUSH("light accumulator")
		if (lc)
            lc->averageImage(frame, detector.H);
                res = true;
        PROFILE_SECTION_POP();
	}

	PROFILE_SECTION_POP();


	return res;
}

bool add_detected_homography(int n, planar_object_recognizer &detector, CamCalibration &calib)
{
	static std::vector<CamCalibration::s_struct_points> pts;
	pts.clear();

	for (int i=0; i<detector.match_number; ++i) {
			image_object_point_match * match = detector.matches+i;
			if (match->inlier) {
				pts.push_back(CamCalibration::s_struct_points(
					PyrImage::convCoordf(match->image_point->u, int(match->image_point->scale), 0),
					PyrImage::convCoordf(match->image_point->v, int(match->image_point->scale), 0),
					PyrImage::convCoordf((float)match->object_point->M[0], int(match->object_point->scale), 0),
					PyrImage::convCoordf((float)match->object_point->M[1], int(match->object_point->scale), 0)));
			}
	}

	return calib.AddHomography(n, pts, detector.H);
}

bool add_detected_homography(int n, planar_object_recognizer &detector, CamAugmentation &a)
{
	static std::vector<CamCalibration::s_struct_points> pts;
	pts.clear();

	for (int i=0; i<detector.match_number; ++i) {
			image_object_point_match * match = detector.matches+i;
			if (match->inlier) {
				pts.push_back(CamCalibration::s_struct_points(
					PyrImage::convCoordf(match->image_point->u, int(match->image_point->scale), 0),
					PyrImage::convCoordf(match->image_point->v, int(match->image_point->scale), 0),
					PyrImage::convCoordf(match->object_point->M[0], int(match->object_point->scale), 0),
					PyrImage::convCoordf(match->object_point->M[1], int(match->object_point->scale), 0)));
			}
	}

	a.AddHomography(pts, detector.H);
	return true;
}
