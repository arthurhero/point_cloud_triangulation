import numpy as np
import os
import time
import sys
from open3d import io,utility,visualization,geometry
import math

PI = math.pi

def read_file(fname):
    pcd = io.read_point_cloud(fname)
    #visualization.draw_geometries([pcd])
    points = np.asarray(pcd.points)
    return points

def view_pc(points):
    '''
    points - n x 3
    '''
    pcd = geometry.PointCloud()
    pcd.points = utility.Vector3dVector(pc)
    '''
    optional color
    pcd.colors = utility.Vector3dVector(color)
    '''
    visualization.draw_geometries([pcd])

def view_mesh(points,faces):
    '''
    points - n x 3
    faces - np.uint16 
    '''
    mesh=geometry.TriangleMesh()
    mesh.vertices=utility.Vector3dVector(points)
    mesh.triangles=utility.Vector3iVector(faces)
    mesh.compute_vertex_normals()
    mesh.paint_uniform_color([1, 0.706, 0])
    visualization.draw_geometries([mesh])

def get_edge(v1, v2):
    return (min(v1, v2), max(v1, v2))

def get_dist(p1,p2):
    return ((p1-p2)**2).sum()**0.5

def ccw(a,b,c):
    # check if counterclockwise
    return (c[1]-a[1]) * (b[0]-a[0]) > (b[1]-a[1]) * (c[0]-a[0])

def check_intersect(p1, p2, q1, q2):
    # check if two line segments on 2d intersect
    return ccw(p1,q1,q2) != ccw(p2,q1,q2) and ccw(p1,p2,q1) != ccw(p1,p2,q2)

def get_neighbors(points, pi, mu, adj_points, face_normals, adj_faces, boundary_edges):
    '''
    points - n x 3
    pi - idx of p
    mu - scalar
    adj_points - k, indices of vertices, already confirmed edges
    face_normals - m x 3, float
    adj_faces, indices of already constructed faces incident to p
    boundary_edges - boundary edges in the entire shape
    return
    nn_idx - ordered by angle in counterclockwise, pruned by distance and visibility 
    '''
    p = points[pi]
    # get minimal distance
    dist_all = ((points-p[None,:])**2).sum(axis=1)**0.5 # n
    if len(adj_points)>0:
        min_dist = None
        for ai in adj_points:
            ap = points[ai]
            dist = get_dist(ap, p)
            if min_dist is None or dist < min_dist:
                min_dist = dist
    else:
        min_dist = np.delete(dist_all, pi).min()
    prune_dist = min_dist * mu

    # prune by distance
    nn_idx = np.flatnonzero(np.logical_and(dist_all > 0, dist_all <= prune_dist))
    nn_p = points[nn_idx]

    # get local plane
    if len(adj_faces) > 0:
        # if have incident triangles, use avg face normal
        avg_normal = np.zeros((3,))
        for fi in adj_faces:
            fn = face_normals[fi]
            avg_normal += fn
        avg_normal /= len(adj_faces)
        normal = avg_normal / np.linalg.norm(avg_normal) 
    else:
        # get fitted plane using pca
        svd = np.linalg.svd(nn_p - np.mean(nn_p, axis=0, keepdims=True))
        normal = svd[2][-1]

    # project to local plane
    nn_p_rel = nn_p-p[None,:] # relative nn position
    basis1 = np.cross(normal, nn_p_rel[0]) # randomly choose a basis
    basis2 = np.cross(normal, basis1)
    basis1 /= np.linalg.norm(basis1)
    basis2 /= np.linalg.norm(basis2)
    nn_p_plane = nn_p_rel - (nn_p_rel @ normal[:,None])*normal[None,:] # points projected onto the plane
    nn_p_uv = np.concatenate([nn_p_plane @ basis1[:,None], nn_p_plane @ basis2[:,None]], axis=1) # n' x 2

    # prune by visibility
    visible = np.ones((len(nn_idx),), dtype=bool)
    # first delete points part of inside edge
    for j in range(len(nn_idx)):
        ni = nn_idx[j]
        if ni.item() in adj_points:
            if get_edge(pi, ni) not in boundary_edges:
                visible[j] = False
    nn_idx = nn_idx[visible]
    nn_p_uv = nn_p_uv[visible]
    visible = np.ones((len(nn_idx),), dtype=bool)
    # then find all boundary edges close to p
    for edge in boundary_edges:
        p1 = points[edge[0]]
        p2 = points[edge[1]]
        if pi == edge[0] or pi == edge[1]:
            continue
        dist = np.linalg.norm(np.cross(p-p1,p-p2)) / np.linalg.norm(p2-p1) # distance from edge to p
        if dist < prune_dist:
            # project edge to plane
            p1_plane = (p1-p) - np.dot((p1-p),normal) * normal
            p1_uv = np.asarray([np.dot(p1_plane,basis1), np.dot(p1_plane,basis2)])
            p2_plane = (p2-p) - np.dot((p2-p),normal) * normal
            p2_uv = np.asarray([np.dot(p2_plane,basis1), np.dot(p2_plane,basis2)])
            for k in range(len(nn_idx)):
                if nn_idx[k] == edge[0] or nn_idx[k] == edge[1]:
                    continue
                np_uv = nn_p_uv[k]
                if visible[k] == True and check_intersect(p1_uv,p2_uv,np.array([0,0]),np_uv): # p in uv-coord is (0,0)
                    #print("intersection! p1uv,p2uv,np_uv,p1,p2",p1_uv,p2_uv,np_uv,edge,nn_idx[k])
                    visible[k] = False
    nn_idx = nn_idx[visible]
    nn_p_uv = nn_p_uv[visible]

    # order the neighbors counterclockwise
    nn_p_rad = np.arctan2(nn_p_uv[:,1], nn_p_uv[:,0])
    sort_idx = np.argsort(nn_p_rad)
    nn_idx = nn_idx[sort_idx]
    nn_p_uv = nn_p_uv[sort_idx]

    return nn_idx, nn_p_uv

def check_cw_face(face, i, ni):
    # check if the face has vertex to the right of vector i,ni
    face_set = set(face)
    assert len(face_set)==3, "invalid face"
    if i in face_set and ni in face_set:
        if face[0]==i:
            return face[2]==ni
        if face[0]==ni:
            return face[1]==i
        if face[1]==ni:
            return face[2]==i
    return False

def triangulate(points, mu):
    '''
    points - n x 3
    mu - scalar
    return
    faces - m x 3, np.uint16, indices of vertices
    '''
    faces = list()
    face_normals = list()
    n = points.shape[0]

    adj_points = [set() for _ in range(n)] # list of sets
    adj_faces = [set() for _ in range(n)]

    is_boundary= dict() # key: edge (vert1, vert2), value: bool 
    boundary_edges = set()

    visited = set()
    frontier = [0] 

    while len(frontier) > 0:
        '''
        if len(faces) > 0:
            view_mesh(points, np.asarray(faces, dtype = np.uint16))
            '''
        i = frontier.pop(0) # bfs
        print('i',i)
        #print('num boundary edge',len(boundary_edges))
        if i in visited:
            continue
        p = points[i]
        nn_idx, nn_p_uv = get_neighbors(points, i, mu, adj_points[i], face_normals, adj_faces[i], boundary_edges)
        if len(nn_idx)>1:
            nn_idx = np.concatenate([nn_idx, nn_idx[0:1]],axis=0) # make it a cycle
            nn_p_uv = np.concatenate([nn_p_uv, nn_p_uv[0:1]],axis=0) # make it a cycle
        # triangulate
        last_edge = None
        last_ni = None
        for j in range(len(nn_idx)):
            ni = nn_idx[j].item()
            # add neighbors to frontier
            if ni not in visited:
                frontier.append(ni)
            npo = points[ni]
            ne = get_edge(ni, i)
            #print("i,ni,uv",i,ni,nn_p_uv[j])
            if len(nn_idx)>1:
                if j<len(nn_idx)-1:
                    assert ne not in is_boundary or is_boundary[ne] == True, "current edge cannot be completed"
                elif ne in is_boundary and is_boundary[ne] == False:
                    continue
            # if it is a new edge, we add to adj list 
            if ni not in adj_points[i]:
                adj_points[i].add(ni)
                adj_points[ni].add(i)
            if last_edge is not None and last_edge in is_boundary and is_boundary[last_edge] == False:
                # if last edge is already completed, skip (usually caused by the neighbor between too far away from p)
                last_edge = ne
                last_ni = ni
                continue
            # if last edge exists and is not completed
            if last_edge is not None and (last_edge not in is_boundary or is_boundary[last_edge] == True):
                # check if the current edge already has a face to the cw side
                # and if last edge has a face to the ccw side
                has_face = False
                for fi in adj_faces[i]:
                    face = faces[fi]
                    if check_cw_face(face, i, ni):
                        has_face = True
                        break
                    if check_cw_face(face, last_ni, i):
                        has_face = True
                        break
                if has_face:
                    last_edge = ne
                    last_ni = ni
                    continue
                # create new face
                nf = [i, last_ni, ni]
                # check if the new face is cw
                if not ccw(np.zeros((2,)),nn_p_uv[j-1],nn_p_uv[j]):
                    last_edge = ne
                    last_ni = ni
                    continue
                # add faces, edges, etc to the lists
                faces.append(nf)
                adj_faces[i].add(len(faces)-1)
                adj_faces[ni].add(len(faces)-1)
                adj_faces[last_ni].add(len(faces)-1)
                fnormal = np.cross(points[last_ni]-p,npo-p) # get face normal
                fnormal /= np.linalg.norm(fnormal) # normalize
                face_normals.append(fnormal)
                oppo_edge = get_edge(last_ni, ni)
                adj_points[ni].add(last_ni)
                adj_points[last_ni].add(ni)
                # check edges still boundary or not
                if oppo_edge in is_boundary:
                    if is_boundary[oppo_edge]==True:
                        is_boundary[oppo_edge] = False
                        boundary_edges.remove(oppo_edge)
                    else:
                        print("dup edge!1")
                else:
                    is_boundary[oppo_edge] = True
                    boundary_edges.add(oppo_edge)
                if last_edge in is_boundary:
                    if is_boundary[last_edge]==True:
                        is_boundary[last_edge] = False
                        boundary_edges.remove(last_edge)
                    else:
                        print("dup edge!2")
                else:
                    is_boundary[last_edge] = True
                    boundary_edges.add(last_edge)
                if ne in is_boundary:
                    if is_boundary[ne]==True:
                        is_boundary[ne] = False
                        boundary_edges.remove(ne)
                    else:
                        print("dup edge!3")
                else:
                    is_boundary[ne] = True
                    boundary_edges.add(ne)
            # update last edge
            last_edge = ne
            last_ni = ni

        visited.add(i)

    return np.asarray(faces, dtype = np.uint16)


if __name__ == "__main__":
    mu = 1.5
    points = read_file("bunny.ply")
    faces = triangulate(points, mu)
    np.save("bunny_faces",faces)
    view_mesh(points, faces)
