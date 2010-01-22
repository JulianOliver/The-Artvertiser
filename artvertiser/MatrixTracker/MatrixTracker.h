#pragma once

#include "ofxMatrix4x4.h"
#include <cv.h>
#include "../FProfiler/FTime.h"
#include "../FProfiler/FSemaphore.h"
#include <map>
#include "KeypointNeighbourSearch.h"

using namespace std;

class MatrixTracker
{
public:

    /// track. matrix is 3x4 with translation in last column, keypoints are keypoints, num_keypoints is count,
    /// corners are the 4 corners of the detected rectangle _in keypoint(2d) space_.
    void addPose( CvMat* matrix, const FTime& timestamp, keypoint* keypoints=NULL, int num_keypoints=0, ofxVec2f* corners=NULL );
    /// track. rotation_matrix is 3x3 rot matrix, trans is translation
    void addPose( const ofxMatrix4x4& rotation_matrix, const ofxVec3f& trans, const FTime& timestamp,
                 keypoint* keypoints=NULL, int num_keypoints=0, ofxVec2f* corners=NULL );
    /// track by finding a keypoint match

    /// Put an interpolated pose for the given timestamp into interpolated_pose, as a 3x4 matrix
    /// with translation in the last column. Return false if a pose couldn't be calculated.
    bool getInterpolatedPose( ofxMatrix4x4& interpolated_pose, const FTime& for_time, ofxVec3f& object_space_delta_since_last_stored );
    /// Put an interpolated pose for the given timestamp into interpolated_pose, as a 3x4 matrix
    /// with translation in the last column. Return false if a pose couldn't be calculated.
    bool getInterpolatedPose( CvMat* matrix, const FTime& for_time, ofxVec3f& object_space_delta_since_last_stored );

    /// Take the given translation estimate in screen space relative and, using the keypoint data from this frame,
    /// refine the estimate against the stored keypoints.
    bool refinePoseUsingKeypoints( const ofxVec2f& translation_2d_estimate_screen_space, keypoint* keypoints, int num_keypoints,
                                 const FTime& timestamp, ofxVec2f& refined_2d_translation_screen_space );

    /// Put an interpolated translation for the given timestamp into interpolated_translation
    bool getInterpolatedTranslation ( ofxVec3f& interpolated_translation, const FTime& for_time );

private:

    ofxVec3f extractTranslation( const ofxMatrix4x4& pose ) { return ofxVec3f(pose(0,3),pose(1,3),pose(2,3)); }

    void lock() { poses_lock.Wait(); };
    void unlock() { poses_lock.Signal(); };

    class Pose {
    public:
        Pose() {};
        Pose( const ofxVec3f& trans, const ofxQuaternion& rot ): translation(trans), rotation(rot) {}
        ofxVec3f translation;
        ofxQuaternion rotation;
    };

    // map allows us to do lower_bound and upper_bound
    typedef map<FTime, Pose> PoseMap;
    PoseMap found_poses;


    ofxVec3f prev_returned_translation;
    ofxQuaternion prev_returned_rotation;

    void make4x4MatrixFromQuatTrans( const ofxQuaternion& rot, const ofxVec3f& trans, ofxMatrix4x4& output );
    void smoothAndMakeMatrix( const ofxQuaternion& rot, const ofxVec3f& trans, ofxMatrix4x4& output );

    FSemaphore poses_lock;

    KeypointNeighbourSearch keypoint_search;

};


