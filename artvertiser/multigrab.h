#ifndef _MULTIGRAB_H
#define _MULTIGRAB_H

#include "calibmodel.h"

#define USE_MULTITHREADCAPTURE

#include "multithreadcapture.h"

class MultiGrab {
public:

	CalibModel model;

	MultiGrab(const char *modelfile="model.bmp") : model(modelfile) {}
    ~MultiGrab();

	int init(bool cacheTraining, char *modelfile, char *avi_bg_path, int capture_width, int capture_height, int v4l_device, int detect_width, int detect_height );
	void grabFrames();
	void allocLightCollector();

	class Cam {
    public:
		CvCapture *cam;
		int width,height;
		int detect_width, detect_height;
		//PlanarObjectDetector detector;
		planar_object_recognizer detector;
		LightCollector *lc;

		/// stop capturing but leave frame buffers in place
		void shutdownMultiThreadCapture();

        const FTime& getLastProcessedFrameTimestamp() { return detected_frame_timestamp; }
		IplImage* getLastProcessedFrame() { return frame; }
		/// fetch a pointer to the last frame that should be used for drawing
		/// internally double buffered: pointers remain valid until the next call to getLastDrawFrame.
		bool getLastDrawFrame( IplImage** draw_frame, FTime* timestamp_copy=NULL,
                        keypoint** keypoints=NULL, int *num_keypoints = NULL )
            { return mtc->getLastDrawFrame( draw_frame, timestamp_copy, keypoints, num_keypoints, /*blocking*/ true ); }

		/// fetch the last raw frame + timestamp and put into *frame + timestamp. if *frame is NULL, create.
		/// see MultiThreadCapture::getCopyOfLastFrame.
		bool getCopyOfLastFrame( IplImage** raw_frame, FTime* timestamp_copy=NULL )
            { return mtc->getCopyOfLastFrame( raw_frame, timestamp_copy, true /*block*/ ); }

		/// fetch a pointer to the pointer to the keypoint detector
		/// must release lock when done
		/// sorry about how ugly this is. see MultiThreadCapture for the origin, planar_object_recognizer for the destination
		/// (see planar_object_recognizer.load() or search for 'new pyr_yape' in planar_object_recognizer.cpp)
        pyr_yape** getKeypointDetectorPtrPtr() { return mtc->getKeypointDetectorPtrPtr(); }
        void releaseKeypointDetectorLock() { mtc->releaseKeypointDetectorLock(); }

		void setCam(CvCapture *c, int capture_width, int capture_height, int detect_width, int detect_height );
		bool detect( keypoint** output_keypoints=NULL, int* num_keypoints=NULL );

		Cam(CvCapture *c=0, int _width=0, int _height=0, int _detect_width=320, int _detect_height=240)
		{
		    frame = 0;
			width=0;
			height=0;
			detect_width=_detect_width;
			detect_height=_detect_height;
			cam=0;
			lc=0;
			mtc=0;
			if (c) setCam(c, _width, _height, _detect_width, _detect_height );
			gray=0;
			frame_detectsize=0;
		}
		Cam( const Cam& other ) { assert( false && "copy constructor called, arrgh" ); }
		~Cam();

		private:
            IplImage *frame, *frame_detectsize, *gray;
            FTime detected_frame_timestamp;
       		MultiThreadCapture *mtc;


	};

	std::vector<Cam *> cams;
	struct Cam *foo;
};

bool add_detected_homography(int n, planar_object_recognizer &detector, CamCalibration &calib);
bool add_detected_homography(int n, planar_object_recognizer &detector, CamAugmentation &a);
IplImage *myQueryFrame(CvCapture *capture);
IplImage *myRetrieveFrame(CvCapture *capture);

#endif
