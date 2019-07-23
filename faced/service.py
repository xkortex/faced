#!/usr/bin/env python

import sys
import os
import time
import argparse


class DummyVidCap(object):
    def __init__(self, *args, **kwargs):
        """Does nothing but provides release()"""
    def __bool__(self):
        return False

    def release(self):
        pass


def load_arguments():
    parser = argparse.ArgumentParser(
        description='Detect faces in images, videos or webcam.')
    parser.add_argument("-i", "--input", type=str,
                        help="Path to image, video or just 'webcam' for live "
                             "detection.")
    parser.add_argument("-t", "--threshold", type=float, default=0.85,
                        help="0 to 1 float number specifying selectivity.")
    parser.add_argument("-s", "--save",
                        help="If this option is specified, result will be "
                             "saved in `output` file. It does not work with "
                             "webcam option.",
                        action='store_true')
    parser.add_argument("-o", "--out_dir", type=str,
                        help="output directory to place generated files",
                        action='store')
    parser.add_argument("-g", "--gui",
                        help="Show the GUI output",
                        action='store_true')
    parser.add_argument("-v", "--verbose",
                        help="Print verbose output",
                        action='store_true')

    args = parser.parse_args()
    return args


def is_video(filename):
    exts = (
            '.264', '.3g2', '.3gp', '.3gp2', '.3gpp', '.3gpp2', '.3mm', '.3p2', '.60d', '.787', '.89', '.aaf', '.aec', '.aep', '.aepx',
            '.aet', '.aetx', '.ajp', '.ale', '.am', '.amc', '.amv', '.amx', '.anim', '.aqt', '.arcut', '.arf', '.asf', '.asx', '.avb',
            '.avc', '.avd', '.avi', '.avp', '.avs', '.avs', '.avv', '.axm', '.bdm', '.bdmv', '.bdt2', '.bdt3', '.bik', '.bin', '.bix',
            '.bmk', '.bnp', '.box', '.bs4', '.bsf', '.bvr', '.byu', '.camproj', '.camrec', '.camv', '.ced', '.cel', '.cine', '.cip',
            '.clpi', '.cmmp', '.cmmtpl', '.cmproj', '.cmrec', '.cpi', '.cst', '.cvc', '.cx3', '.d2v', '.d3v', '.dat', '.dav', '.dce',
            '.dck', '.dcr', '.dcr', '.ddat', '.dif', '.dir', '.divx', '.dlx', '.dmb', '.dmsd', '.dmsd3d', '.dmsm', '.dmsm3d', '.dmss',
            '.dmx', '.dnc', '.dpa', '.dpg', '.dream', '.dsy', '.dv', '.dv-avi', '.dv4', '.dvdmedia', '.dvr', '.dvr-ms', '.dvx', '.dxr',
            '.dzm', '.dzp', '.dzt', '.edl', '.evo', '.eye', '.ezt', '.f4p', '.f4v', '.fbr', '.fbr', '.fbz', '.fcp', '.fcproject',
            '.ffd', '.flc', '.flh', '.fli', '.flv', '.flx', '.gfp', '.gl', '.gom', '.grasp', '.gts', '.gvi', '.gvp', '.h264', '.hdmov',
            '.hkm', '.ifo', '.imovieproj', '.imovieproject', '.ircp', '.irf', '.ism', '.ismc', '.ismv', '.iva', '.ivf', '.ivr', '.ivs',
            '.izz', '.izzy', '.jss', '.jts', '.jtv', '.k3g', '.kmv', '.ktn', '.lrec', '.lsf', '.lsx', '.m15', '.m1pg', '.m1v', '.m21',
            '.m21', '.m2a', '.m2p', '.m2t', '.m2ts', '.m2v', '.m4e', '.m4u', '.m4v', '.m75', '.mani', '.meta', '.mgv', '.mj2', '.mjp',
            '.mjpg', '.mk3d', '.mkv', '.mmv', '.mnv', '.mob', '.mod', '.modd', '.moff', '.moi', '.moov', '.mov', '.movie', '.mp21',
            '.mp21', '.mp2v', '.mp4', '.mp4v', '.mpe', '.mpeg', '.mpeg1', '.mpeg4', '.mpf', '.mpg', '.mpg2', '.mpgindex', '.mpl',
            '.mpl', '.mpls', '.mpsub', '.mpv', '.mpv2', '.mqv', '.msdvd', '.mse', '.msh', '.mswmm', '.mts', '.mtv', '.mvb', '.mvc',
            '.mvd', '.mve', '.mvex', '.mvp', '.mvp', '.mvy', '.mxf', '.mxv', '.mys', '.ncor', '.nsv', '.nut', '.nuv', '.nvc', '.ogm',
            '.ogv', '.ogx', '.osp', '.otrkey', '.pac', '.par', '.pds', '.pgi', '.photoshow', '.piv', '.pjs', '.playlist', '.plproj',
            '.pmf', '.pmv', '.pns', '.ppj', '.prel', '.pro', '.prproj', '.prtl', '.psb', '.psh', '.pssd', '.pva', '.pvr', '.pxv',
            '.qt', '.qtch', '.qtindex', '.qtl', '.qtm', '.qtz', '.r3d', '.rcd', '.rcproject', '.rdb', '.rec', '.rm', '.rmd', '.rmd',
            '.rmp', '.rms', '.rmv', '.rmvb', '.roq', '.rp', '.rsx', '.rts', '.rts', '.rum', '.rv', '.rvid', '.rvl', '.sbk', '.sbt',
            '.scc', '.scm', '.scm', '.scn', '.screenflow', '.sec', '.sedprj', '.seq', '.sfd', '.sfvidcap', '.siv', '.smi', '.smi',
            '.smil', '.smk', '.sml', '.smv', '.spl', '.sqz', '.srt', '.ssf', '.ssm', '.stl', '.str', '.stx', '.svi', '.swf', '.swi',
            '.swt', '.tda3mt', '.tdx', '.thp', '.tivo', '.tix', '.tod', '.tp', '.tp0', '.tpd', '.tpr', '.trp', '.ts', '.tsp', '.ttxt',
            '.tvs', '.usf', '.usm', '.vc1', '.vcpf', '.vcr', '.vcv', '.vdo', '.vdr', '.vdx', '.veg','.vem', '.vep', '.vf', '.vft',
            '.vfw', '.vfz', '.vgz', '.vid', '.video', '.viewlet', '.viv', '.vivo', '.vlab', '.vob', '.vp3', '.vp6', '.vp7', '.vpj',
            '.vro', '.vs4', '.vse', '.vsp', '.w32', '.wcp', '.webm', '.wlmp', '.wm', '.wmd', '.wmmp', '.wmv', '.wmx', '.wot', '.wp3',
            '.wpl', '.wtv', '.wve', '.wvx', '.xej', '.xel', '.xesc', '.xfl', '.xlmv', '.xmv', '.xvid', '.y4m', '.yog', '.yuv', '.zeg',
            '.zm1', '.zm2', '.zm3', '.zmv'  
            )
    return any((filename.endswith(x) for x in exts))


def run_img(path, thresh, save=False, out_dir=None, gui=False, verbose=False):
    import cv2
    import json
    from faced.detector import FaceDetector
    from faced.utils import bboxes_jsonable

    file_out_path = os.path.join(out_dir or '/tmp', os.path.basename(path))

    face_detector = FaceDetector()

    img = cv2.imread(path)
    rgb_img = cv2.cvtColor(img.copy(), cv2.COLOR_BGR2RGB)

    bboxes = face_detector.predict(rgb_img, thresh)
    bboxes = bboxes_jsonable(bboxes)
    if verbose:
        print(bboxes)

    if out_dir:
        with open(file_out_path + '.jtxt', 'a') as fp:
            fp.write(json.dumps(bboxes) + '\n')
            print('written to {}'.format(file_out_path))

    if save or gui:
        from faced.utils import annotate_image
        ann_img = annotate_image(img, bboxes)
        if save:
            cv2.imwrite(file_out_path + ".png", ann_img)
        if gui:
            cv2.imshow('image', ann_img)
            cv2.startWindowThread()
            cv2.waitKey(0)
            cv2.destroyAllWindows()

    return bboxes


def run_video(path, thresh, save=False, out_dir=None, gui=False, verbose=False):
    import cv2

    from faced.detector import FaceDetector
    from faced.utils import bboxes_jsonable

    import json

    file_out_path = os.path.join(out_dir or '/tmp', os.path.basename(path))

    face_detector = FaceDetector()

    if path == 'webcam':
        # From webcam
        cap = cv2.VideoCapture(0)
    else:
        cap = cv2.VideoCapture(path)

    if save:
        width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)   # float
        height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT) # float
        fps = cap.get(cv2.CAP_PROP_FPS) # float

        fourcc = cv2.VideoWriter_fourcc(*'MJPG')
        out_vid = cv2.VideoWriter(file_out_path + '.avi', fourcc, fps,
                              (int(width), int(height)))
    else:
        out_vid = DummyVidCap()

    now = time.time()
    while cap.isOpened():
        now = time.time()
        # Capture frame-by-frame
        ret, frame = cap.read()
        frame_num = cap.get(cv2.CAP_PROP_POS_FRAMES)

        if frame is None:
            break
        if frame.shape[0] == 0:
            break

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        bboxes = face_detector.predict(rgb_frame, thresh)
        bboxes = bboxes_jsonable(bboxes)

        if out_dir:
            with open(file_out_path + '.jtxt', 'a') as fp:
                fp.write(json.dumps(bboxes) + '\n')

        if save or gui:
            from faced.utils import annotate_image
            ann_frame = annotate_image(frame, bboxes)
            if save:
                out_vid.write(frame)
            if gui:
                cv2.imshow('window', ann_frame)

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break


        print("FPS: {:0.2f}".format(1 / (time.time() - now)), end="\r", flush=True)

    # When everything done, release the capture
    cap.release()
    out_vid.release()


if __name__ == "__main__":
    args = load_arguments()

    if not args.input:
        print("Should specify input. See faced -h for instructions.")
        sys.exit(1)

    if args.input == "webcam" and args.save:
        print("Cannot save if reading from webcam.")
        sys.exit(1)


    path = args.input
    t = None if not args.threshold else float(args.threshold)

    if is_video(path) or path == 'webcam':
        run_video(path, t, args.save, args.out_dir, args.gui, args.verbose)
    else:
        run_img(path, t, args.save, args.out_dir, args.gui, args.verbose)
