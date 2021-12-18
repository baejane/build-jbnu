import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import cv2

from camera import Camera


############ essential matrix#########################################################
def compute_essential_normalized(p1, p2):
    return compute_normalized_image_to_image_matrix(p1, p2, compute_essential=True)

def correspondence_matrix(p1, p2):
    p1x, p1y = p1[:2]
    p2x, p2y = p2[:2]

    return np.array([
        p1x * p2x, p1x * p2y, p1x,
        p1y * p2x, p1y * p2y, p1y,
        p2x, p2y, np.ones(len(p1x))
    ]).T

    return np.array([
        p2x * p1x, p2x * p1y, p2x,
        p2y * p1x, p2y * p1y, p2y,
        p1x, p1y, np.ones(len(p1x))
    ]).T


def compute_image_to_image_matrix(x1, x2, compute_essential=False):
    """ Compute the fundamental or essential matrix from corresponding points
        (x1, x2 3*n arrays) using the 8 point algorithm.
        Each row in the A matrix below is constructed as
        [x'*x, x'*y, x', y'*x, y'*y, y', x, y, 1]
    """
    A = correspondence_matrix(x1, x2)
    # compute linear least square solution
    U, S, V = np.linalg.svd(A)
    F = V[-1].reshape(3, 3)

    # constrain F. Make rank 2 by zeroing out last singular value
    U, S, V = np.linalg.svd(F)
    S[-1] = 0
    if compute_essential:
        S = [1, 1, 0] # Force rank 2 and equal eigenvalues
    F = np.dot(U, np.dot(np.diag(S), V))

    return F

def scale_and_translate_points(points):
    """ Scale and translate image points so that centroid of the points
        are at the origin and avg distance to the origin is equal to sqrt(2).
    :param points: array of homogenous point (3 x n)
    :returns: array of same input shape and its normalization matrix
    """
    x = points[0]
    y = points[1]
    center = points.mean(axis=1)  # mean of each row
    cx = x - center[0] # center the points
    cy = y - center[1]
    dist = np.sqrt(np.power(cx, 2) + np.power(cy, 2))
    scale = np.sqrt(2) / dist.mean()
    norm3d = np.array([
        [scale, 0, -scale * center[0]],
        [0, scale, -scale * center[1]],
        [0, 0, 1]
    ])

    return np.dot(norm3d, points), norm3d



def compute_normalized_image_to_image_matrix(p1, p2, compute_essential=False):
    """ Computes the fundamental or essential matrix from corresponding points
        using the normalized 8 point algorithm.
    :input p1, p2: corresponding points with shape 3 x n
    :returns: fundamental or essential matrix with shape 3 x 3
    """
    n = p1.shape[1]
    if p2.shape[1] != n:
        raise ValueError('Number of points do not match.')

    # preprocess image coordinates
    p1n, T1 = scale_and_translate_points(p1)
    p2n, T2 = scale_and_translate_points(p2)

    # compute F or E with the coordinates
    F = compute_image_to_image_matrix(p1n, p2n, compute_essential)

    # reverse preprocessing of coordinates
    # We know that P1' E P2 = 0
    F = np.dot(T1.T, np.dot(F, T2))

    return F / F[2, 2]
#########################################################################


#특징점 추출 후 correspoing
def find_correspondence_points(img1, img2):
    #harris=특징점 추출
    sift = cv2.xfeatures2d.SIFT_create()

    # 특징점 추출
    kp1, des1 = sift.detectAndCompute(
        cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY), None)
    kp2, des2 = sift.detectAndCompute(
        cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY), None)


    #특징점 매칭 - BFMatcher
    
    
    brute_force_match = cv2.BFMatcher()
    matches = brute_force_match.knnMatch(des1, des2, k=2)

    good = []
    for m, n in matches:
        if m.distance < 0.8 * n.distance:
            good.append(m)

    src_pts = np.asarray([kp1[m.queryIdx].pt for m in good])
    dst_pts = np.asarray([kp2[m.trainIdx].pt for m in good])

    # Constrain matches to fit homography
    retval, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 100.0)
    mask = mask.ravel()

    # We select only inlier points
    pts1 = src_pts[mask == 1]
    pts2 = dst_pts[mask == 1]

    return pts1.T, pts2.T

#직교좌표에다가 1차원추가 
def cart2hom(arr):
    if arr.ndim == 1:
        return np.hstack([arr, 1])
    return np.asarray(np.vstack([arr, np.ones(arr.shape[1])]))

# 2번째 카메라 matrix계산 E = [t]R 이용, E=U시그마Vt
def compute_P_from_essential(E):
    #svd함수를 이용해 반환값 
    U, S, V = np.linalg.svd(E)

    # Ensure rotation matrix are right-handed with positive determinant
    if np.linalg.det(np.dot(U, V)) < 0:
        V = -V

    # 4개의 가능성있는 matrix return (Hartley p 258)
    W = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]])
    P2s = [np.vstack((np.dot(U, np.dot(W, V)).T, U[:, 2])).T,
          np.vstack((np.dot(U, np.dot(W, V)).T, -U[:, 2])).T,
          np.vstack((np.dot(U, np.dot(W.T, V)).T, U[:, 2])).T,
          np.vstack((np.dot(U, np.dot(W.T, V)).T, -U[:, 2])).T]

    return P2s

##############camera matrix 찾기###################3
def reconstruct_one_point(pt1, pt2, m1, m2):
    """
        pt1 and m1 * X are parallel and cross product = 0
        pt1 x m1 * X  =  pt2 x m2 * X  =  0
    """
    A = np.vstack([
        np.dot(skew(pt1), m1),
        np.dot(skew(pt2), m2)
    ])
    U, S, V = np.linalg.svd(A)
    P = np.ravel(V[-1, :4])

    return P / P[3]


def skew(x):
    """ Create a skew symmetric matrix *A* from a 3d vector *x*.
        Property: np.cross(A, v) == np.dot(x, v)
    :param x: 3d vector
    :returns: 3 x 3 skew symmetric matrix from *x*
    """
    return np.array([
        [0, -x[2], x[1]],
        [x[2], 0, -x[0]],
        [-x[1], x[0], 0]
    ])

########################################################

#3D포인터로 변환
def linear_triangulation(p1, p2, m1, m2):
    """
    Linear triangulation (Hartley ch 12.2 pg 312) to find the 3D point X
    where p1 = m1 * X and p2 = m2 * X. Solve AX = 0.
    :param p1, p2: 2D points in homo. or catesian coordinates. Shape (3 x n)
    :param m1, m2: Camera matrices associated with p1 and p2. Shape (3 x 4)
    :returns: 4 x n homogenous 3d triangulated points
    """
    num_points = p1.shape[1]
    res = np.ones((4, num_points))

    for i in range(num_points):
        A = np.asarray([
            (p1[0, i] * m1[2, :] - m1[0, :]),
            (p1[1, i] * m1[2, :] - m1[1, :]),
            (p2[0, i] * m2[2, :] - m2[0, :]),
            (p2[1, i] * m2[2, :] - m2[1, :])
        ])

        _, _, V = np.linalg.svd(A)
        X = V[-1, :4]
        res[:, i] = X / X[3]

    return res


def dino():
    #1. image 불러오기
    img1 = cv2.imread('imagesit/1.jpg') 
    img2 = cv2.imread('imagesit/2.jpg')
    img3 = cv2.imread('imagesit/3.jpg')
    img4 = cv2.imread('imagesit/4.jpg')
    img5 = cv2.imread('imagesit/5.jpg')
    img6 = cv2.imread('imagesit/6.jpg')
    img7 = cv2.imread('imagesit/7.jpg')
    img8 = cv2.imread('imagesit/8.jpg')
    img9 = cv2.imread('imagesit/9.jpg')
    img10 = cv2.imread('imagesit/10.jpg')

    #feature matches 찾기
    pts1, pts2 = find_correspondence_points(img1, img2) #앞쪽
    pts3, pts4 = find_correspondence_points(img2, img3)
    pts5, pts6 = find_correspondence_points(img3, img4)
    
    pts13, pts14 = find_correspondence_points(img7, img8) #뒷쪽
    pts15, pts16 = find_correspondence_points(img8, img9)
    pts17, pts18 = find_correspondence_points(img9, img10)

    #직교좌표계에서 homogenouse로 변경
    points1 = cart2hom(pts1) 
    points2 = cart2hom(pts2)
    points3 = cart2hom(pts3)
    points4 = cart2hom(pts4)
    points5 = cart2hom(pts5)
    points6 = cart2hom(pts6)
    
    
    points13 = cart2hom(pts13)
    points14 = cart2hom(pts14)
    points15 = cart2hom(pts15)
    points16 = cart2hom(pts16)
    points17 = cart2hom(pts17)
    points18 = cart2hom(pts18)
    
    
    #사진 띄우기
    fig, ax = plt.subplots(5, 2)
    ax[0][0].autoscale_view('tight')
    ax[0][0].imshow(cv2.cvtColor(img1, cv2.COLOR_BGR2RGB))
    ax[0][0].plot(points1[0], points1[1], 'r.')

    ax[0][1].autoscale_view('tight')
    ax[0][1].imshow(cv2.cvtColor(img2, cv2.COLOR_BGR2RGB))
    ax[0][1].plot(points2[0], points2[1], 'r.')

    ax[1][0].autoscale_view('tight')
    ax[1][0].imshow(cv2.cvtColor(img3, cv2.COLOR_BGR2RGB))
    ax[1][0].plot(points2[0], points2[1], 'r.')

    ax[1][1].autoscale_view('tight')
    ax[1][1].imshow(cv2.cvtColor(img4, cv2.COLOR_BGR2RGB))
    ax[1][1].plot(points2[0], points2[1], 'r.')

    ax[2][0].autoscale_view('tight')
    ax[2][0].imshow(cv2.cvtColor(img5, cv2.COLOR_BGR2RGB))
    ax[2][0].plot(points2[0], points2[1], 'r.')

    ax[2][1].autoscale_view('tight')
    ax[2][1].imshow(cv2.cvtColor(img6, cv2.COLOR_BGR2RGB))
    ax[2][1].plot(points2[0], points2[1], 'r.')

    ax[3][0].autoscale_view('tight')
    ax[3][0].imshow(cv2.cvtColor(img7, cv2.COLOR_BGR2RGB))
    ax[3][0].plot(points2[0], points2[1], 'r.')

    ax[3][1].autoscale_view('tight')
    ax[3][1].imshow(cv2.cvtColor(img8, cv2.COLOR_BGR2RGB))
    ax[3][1].plot(points2[0], points2[1], 'r.')

    ax[4][0].autoscale_view('tight')
    ax[4][0].imshow(cv2.cvtColor(img9, cv2.COLOR_BGR2RGB))
    ax[4][0].plot(points2[0], points2[1], 'r.')

    ax[4][1].autoscale_view('tight')
    ax[4][1].imshow(cv2.cvtColor(img10, cv2.COLOR_BGR2RGB))
    ax[4][1].plot(points2[0], points2[1], 'r.')
    
    fig.show()

    #내부 파라미터 
    height, width, ch = img1.shape
    intrinsic = np.array([  
        [2360, 0, width / 2],
        [0, 2360, height / 2],
        [0, 0, 1]])

    return points1, points2,points3,points4,points5,points6,points13,points14,points15,points16,points17,points18, intrinsic
  
points1, points2, points3,points4,points5,points6,points13,points14,points15,points16,points17,points18, intrinsic = dino()



#내부파라미터*feautre벡터
points1n = np.dot(np.linalg.inv(intrinsic), points1)
points2n = np.dot(np.linalg.inv(intrinsic), points2)
points3n = np.dot(np.linalg.inv(intrinsic), points3)
points4n = np.dot(np.linalg.inv(intrinsic), points4)
points5n = np.dot(np.linalg.inv(intrinsic), points5)
points6n = np.dot(np.linalg.inv(intrinsic), points6)

points13n = np.dot(np.linalg.inv(intrinsic), points13)
points14n = np.dot(np.linalg.inv(intrinsic), points14)
points15n = np.dot(np.linalg.inv(intrinsic), points15)
points16n = np.dot(np.linalg.inv(intrinsic), points16)
points17n = np.dot(np.linalg.inv(intrinsic), points17)
points18n = np.dot(np.linalg.inv(intrinsic), points18)

#essential matrix는 카메라들 사이의 관계를 나타내는 matrix, essential matrix구하기
E = compute_essential_normalized(points1n, points2n)
E2 = compute_essential_normalized(points3n, points4n)
E3 = compute_essential_normalized(points5n, points6n)
E7 = compute_essential_normalized(points13n, points14n)
E8 = compute_essential_normalized(points15n, points16n)
E9 = compute_essential_normalized(points17n, points18n)




#1번째 사진에서 본 2번째 사진의 카메라 파라미터 구하기
P1 = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0]])
P2s = compute_P_from_essential(E)

ind = -1
for i, P2 in enumerate(P2s):
    
    d1 = reconstruct_one_point(
        points1n[:, 0], points2n[:, 0], P1, P2)

    # 카메라 좌표에서 월드 좌표로 변환
    P2_homogenous = np.linalg.inv(np.vstack([P2, [0, 0, 0, 1]]))
    d2 = np.dot(P2_homogenous[:3, :4], d1)

    if d1[2] > 0 and d2[2] > 0:
        ind = i

P2 = np.linalg.inv(np.vstack([P2s[ind], [0, 0, 0, 1]]))[:3, :4]
tripoints3d = linear_triangulation(points1n, points2n, P1, P2)

#######new3,4#####
P3 = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0]])
P4s = compute_P_from_essential(E2)

#ind = -1
for i, P4 in enumerate(P4s):
    
    d1 = reconstruct_one_point(
        points3n[:, 0], points4n[:, 0], P3, P4)

    
    P4_homogenous = np.linalg.inv(np.vstack([P4, [0, 0, 0, 1]]))
    d2 = np.dot(P4_homogenous[:3, :4], d1)

    if d1[2] > 0 and d2[2] > 0:
        ind = i

P4 = np.linalg.inv(np.vstack([P4s[ind], [0, 0, 0, 1]]))[:3, :4]
tripoints3d2 = linear_triangulation(points3n, points4n, P3, P4) 


#######new5,6#####
P5 = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0]])
P6s = compute_P_from_essential(E3)

ind = -1
for i, P6 in enumerate(P6s):
   
    d1 = reconstruct_one_point(
        points5n[:, 0], points6n[:, 0], P5, P6)

    
    P6_homogenous = np.linalg.inv(np.vstack([P6, [0, 0, 0, 1]]))
    d2 = np.dot(P6_homogenous[:3, :4], d1)

    if d1[2] > 0 and d2[2] > 0:
        ind = i

P6 = np.linalg.inv(np.vstack([P6s[ind], [0, 0, 0, 1]]))[:3, :4]
tripoints3d3 = linear_triangulation(points5n, points6n, P5, P6) 




###new13,14####
P13 = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0]])
P14s = compute_P_from_essential(E7)

ind = -1
for i, P14 in enumerate(P14s):    
    d1 = reconstruct_one_point(
        points13n[:, 0], points14n[:, 0], P13, P14)
   
    P14_homogenous = np.linalg.inv(np.vstack([P14, [0, 0, 0, 1]]))
    d2 = np.dot(P14_homogenous[:3, :4], d1)

    if d1[2] > 0 and d2[2] > 0:
        ind = i

P14 = np.linalg.inv(np.vstack([P14s[ind], [0, 0, 0, 1]]))[:3, :4]
tripoints3d7 = linear_triangulation(points13n, points14n, P13, P14)
###new15,16####
P15 = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0]])
P16s = compute_P_from_essential(E8)

ind = -1
for i, P16 in enumerate(P16s):    
    d1 = reconstruct_one_point(
        points15n[:, 0], points16n[:, 0], P15, P16)
   
    P16_homogenous = np.linalg.inv(np.vstack([P16, [0, 0, 0, 1]]))
    d2 = np.dot(P16_homogenous[:3, :4], d1)

    if d1[2] > 0 and d2[2] > 0:
        ind = i

P16 = np.linalg.inv(np.vstack([P16s[ind], [0, 0, 0, 1]]))[:3, :4]
tripoints3d8 = linear_triangulation(points15n, points16n, P15, P16)

###new17,18####
P17 = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0]])
P18s = compute_P_from_essential(E9)

ind = -1
for i, P18 in enumerate(P18s):    
    d1 = reconstruct_one_point(
        points17n[:, 0], points18n[:, 0], P17, P18)
   
    P18_homogenous = np.linalg.inv(np.vstack([P18, [0, 0, 0, 1]]))
    d2 = np.dot(P18_homogenous[:3, :4], d1)

    if d1[2] > 0 and d2[2] > 0:
        ind = i

P18 = np.linalg.inv(np.vstack([P18s[ind], [0, 0, 0, 1]]))[:3, :4]
tripoints3d9 = linear_triangulation(points17n, points18n, P17, P18)



fig = plt.figure()
fig.suptitle('3D reconstructed', fontsize=16)
ax = fig.gca(projection='3d')
ax.plot(tripoints3d[0], tripoints3d[1], tripoints3d[2], 'b.')
ax.plot(tripoints3d2[0], tripoints3d2[1], tripoints3d2[2], 'b.')
ax.plot(tripoints3d3[0], tripoints3d3[1], tripoints3d3[2], 'b.')

ax.plot(tripoints3d7[0], tripoints3d7[1], tripoints3d7[2], 'b.')
ax.plot(tripoints3d8[0], tripoints3d8[1], tripoints3d8[2], 'b.')
ax.plot(tripoints3d9[0], tripoints3d9[1], tripoints3d9[2], 'b.')
ax.set_xlabel('x axis')
ax.set_ylabel('y axis')
ax.set_zlabel('z axis')
ax.view_init(elev=135, azim=90)
plt.show()   
    
