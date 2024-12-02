#include <stdlib.h>
#include <stdint.h> // For uint8_t
#include <math.h>
#include <stdbool.h>
#include <stdio.h>

// Define a struct for a 3D point
typedef struct {
    int x, y, z;
} Point3D;

// Bresenham3D function: Accepts a uint8_t voxel map
Point3D* Bresenham3D(
    int x_ini, int y_ini, int z_ini,
    int *num_points,
    uint8_t *voxel_map, int submap_x, int submap_y, int submap_z,
    Point3D *rays, int num_rays
){
    int max_points = *num_points;
    Point3D *points = (Point3D*)malloc(max_points * sizeof(Point3D));
    if (!points) return NULL; // Check for memory allocation failure

    *num_points = 0; // Initialize the number of points

    // Iterate over the rays
    for (int i = 0; i < num_rays; i++){
        Point3D ray = rays[i];
        int x1 = x_ini, y1 = y_ini, z1 = z_ini;
        int x2 = ray.x+x1, y2 = ray.y+y1, z2 = ray.z+z1;
        int dx = abs(x2 - x1);
        int dy = abs(y2 - y1);
        int dz = abs(z2 - z1);

        int xs = (x2 > x1) ? 1 : -1;
        int ys = (y2 > y1) ? 1 : -1;
        int zs = (z2 > z1) ? 1 : -1;


        // Driving axis is X-axis
        if (dx >= dy && dx >= dz) {
            int p1 = 2 * dy - dx;
            int p2 = 2 * dz - dx;

            bool stop_ray = false;
            while (x1 != x2 && !stop_ray) {
                x1 += xs;
                int move_y = 0, move_z = 0;
                if (p1 >= 0) {
                    move_y = 1;
                    y1 += ys;
                    p1 -= 2 * dx;
                }
                if (p2 >= 0) {
                    move_z = 1;
                    z1 += zs;
                    p2 -= 2 * dx;
                }
                p1 += 2 * dy;
                p2 += 2 * dz;
                if (!(x1 < 0 || x1 >= submap_x || y1 < 0 || y1 >= submap_y || z1 < 0 || z1 >= submap_z)){
                if (move_y && move_z && (voxel_map[(x1 - xs) * submap_y * submap_z + y1 * submap_z + z1] != 0 || voxel_map[x1* submap_y * submap_z + ( y1-ys ) * submap_z +z1] !=0 || voxel_map[x1* submap_y * submap_z + y1*submap_z + (z1 - zs)]!=0 ||
                voxel_map[(x1 - xs) * submap_y * submap_z + (y1 - ys) * submap_z + z1] != 0 || voxel_map[x1 * submap_y * submap_z + (y1-ys) * submap_z + z1 - zs] != 0 || voxel_map[(x1-xs)*submap_y*submap_z + y1*submap_z + z1- zs ] !=0 )){
                    points[(*num_points)++] = (Point3D){(x1 - xs), y1, z1};
                    points[(*num_points)++] = (Point3D){x1, (y1 - ys), z1};
                    points[(*num_points)++] = (Point3D){x1, y1, (z1 - zs)};
                    points[(*num_points)++] = (Point3D){(x1 - xs), (y1 - ys), z1};
                    points[(*num_points)++] = (Point3D){(x1 - xs), y1, (z1 - zs)};
                    points[(*num_points)++] = (Point3D){x1, (y1 - ys), (z1 - zs)};
                    stop_ray = true;
                }
                else if (move_y && !move_z && (voxel_map[(x1 - xs) * submap_y * submap_z + y1 * submap_z + z1] != 0 || voxel_map[x1 * submap_y * submap_z + (y1-ys) * submap_z + z1] != 0)){
                    points[(*num_points)++] = (Point3D){x1, (y1 - ys), z1};
                    points[(*num_points)++] = (Point3D){(x1 -xs) , y1, z1};
                    stop_ray = true;
                }
                else if(move_z && !move_y && (voxel_map[(x1 - xs) * submap_y * submap_z + y1 * submap_z + z1] != 0 || voxel_map[x1 * submap_y * submap_z + y1 * submap_z + z1 - zs] != 0)) {
                    points[(*num_points)++] = (Point3D){(x1 - xs), y1, z1};
                    points[(*num_points)++] = (Point3D){x1, y1, (z1 - zs)};
                    stop_ray = true;
                }
                points[(*num_points)++] = (Point3D){x1, y1, z1};
                stop_ray = voxel_map[x1 * submap_y * submap_z + y1 * submap_z + z1] != 0;}
                else stop_ray = true;
            }
        }
        // Driving axis is Y-axis
        else if (dy >= dx && dy >= dz) {
            int p1 = 2 * dx - dy;
            int p2 = 2 * dz - dy;
            bool stop_ray = false;
            while (y1 != y2 && !stop_ray) {
                y1 += ys;
                int move_x = 0, move_z = 0;
                if (p1 >= 0) {
                    move_x = 1;
                    x1 += xs;
                    p1 -= 2 * dy;
                }
                if (p2 >= 0) {
                    move_z = 1;
                    z1 += zs;
                    p2 -= 2 * dy;
                }
                p1 += 2 * dx;
                p2 += 2 * dz;
                if (!(x1 < 0 || x1 >= submap_x || y1 < 0 || y1 >= submap_y || z1 < 0 || z1 >= submap_z)){
                if (move_x && move_z && (voxel_map[(x1 - xs) * submap_y * submap_z + y1 * submap_z + z1] != 0 || voxel_map[x1 * submap_y * submap_z + (y1 - ys) * submap_z + z1] != 0 || voxel_map[x1 * submap_y * submap_z + y1 * submap_z + (z1 - zs)] != 0 ||
                voxel_map[(x1 - xs) * submap_y * submap_z + (y1 - ys) * submap_z + z1] != 0 || voxel_map[x1 * submap_y * submap_z + (y1 - ys) * submap_z + z1 - zs] != 0 || voxel_map[(x1 - xs) * submap_y * submap_z + y1 * submap_z + z1 - zs] != 0)){
                    points[(*num_points)++] = (Point3D){(x1 - xs), y1, z1};
                    points[(*num_points)++] = (Point3D){x1, (y1 - ys), z1};
                    points[(*num_points)++] = (Point3D){x1, y1, (z1 - zs)};
                    points[(*num_points)++] = (Point3D){(x1 - xs), (y1 - ys), z1};
                    points[(*num_points)++] = (Point3D){(x1 - xs), y1, (z1 - zs)};
                    points[(*num_points)++] = (Point3D){x1, (y1 - ys), (z1 - zs)};
                    stop_ray = true;
                    }
                else if(move_x && !move_z && (voxel_map[(x1 - xs) * submap_y * submap_z + y1 * submap_z + z1] != 0 || voxel_map[x1 * submap_y * submap_z + (y1 - ys) * submap_z + z1] != 0)){
                    points[(*num_points)++] = (Point3D){(x1 - xs), y1, z1};
                    points[(*num_points)++] = (Point3D){x1, (y1 - ys), z1};
                    stop_ray = true;
                }
                else if(move_z && !move_x && (voxel_map[x1 * submap_y * submap_z + (y1-ys) * submap_z + z1] != 0 || voxel_map[x1 * submap_y * submap_z + y1 * submap_z + z1 - zs] != 0)){
                    points[(*num_points)++] = (Point3D){x1, (y1 - ys), z1};
                    points[(*num_points)++] = (Point3D){x1, y1, (z1 - zs)};
                    stop_ray = true;
                }
                points[(*num_points)++] = (Point3D){x1, y1, z1};
                stop_ray = voxel_map[x1 * submap_y * submap_z + y1 * submap_z + z1] != 0;}
                else stop_ray = true;
            }
        }
        // Driving axis is Z-axis
        else {
            int p1 = 2 * dy - dz;
            int p2 = 2 * dx - dz;
            bool stop_ray = false;
            while (z1 != z2 && !stop_ray) {
                z1 += zs;
                int move_x = 0, move_y = 0;
                if (p1 >= 0) {
                    move_y = 1; 
                    y1 += ys;
                    p1 -= 2 * dz;
                }
                if (p2 >= 0) {
                    move_x = 1;
                    x1 += xs;
                    p2 -= 2 * dz;
                }
                p1 += 2 * dy;
                p2 += 2 * dx;
                if (!(x1 < 0 || x1 >= submap_x || y1 < 0 || y1 >= submap_y || z1 < 0 || z1 >= submap_z)){
                if(move_x && move_y && (voxel_map[(x1 - xs) * submap_y * submap_z + y1 * submap_z + z1] != 0 || voxel_map[x1 * submap_y * submap_z + (y1 - ys) * submap_z + z1] != 0 || voxel_map[x1 * submap_y * submap_z + y1 * submap_z + (z1 - zs)] != 0 ||
                voxel_map[(x1 - xs) * submap_y * submap_z + (y1 - ys) * submap_z + z1] != 0 || voxel_map[x1 * submap_y * submap_z + (y1 - ys) * submap_z + z1 - zs] != 0 || voxel_map[(x1 - xs) * submap_y * submap_z + y1 * submap_z + z1 - zs] != 0)){
                    points[(*num_points)++] = (Point3D){(x1 - xs), y1, z1};
                    points[(*num_points)++] = (Point3D){x1, (y1 - ys), z1};
                    points[(*num_points)++] = (Point3D){x1, y1, (z1 - zs)};
                    points[(*num_points)++] = (Point3D){(x1 - xs), (y1 - ys), z1};
                    points[(*num_points)++] = (Point3D){(x1 - xs), y1, (z1 - zs)};
                    points[(*num_points)++] = (Point3D){x1, (y1 - ys), (z1 - zs)};
                    stop_ray = true;
                }
                else if(move_x && !move_y && (voxel_map[(x1 - xs) * submap_y * submap_z + y1 * submap_z + z1] != 0 || voxel_map[x1 * submap_y * submap_z + y1 * submap_z + z1 - zs] != 0)){
                    points[(*num_points)++] = (Point3D){(x1 - xs), y1, z1};
                    points[(*num_points)++] = (Point3D){x1, y1, (z1 - zs)};
                    stop_ray = true;
                }
                else if(move_y && !move_x && (voxel_map[x1 * submap_y * submap_z + (y1 - ys) * submap_z + z1] != 0 || voxel_map[x1 * submap_y * submap_z + y1 * submap_z + z1 - zs] != 0)){
                    points[(*num_points)++] = (Point3D){x1, (y1 - ys), z1};
                    points[(*num_points)++] = (Point3D){x1, y1, (z1 - zs)};
                    stop_ray = true;
                }
                points[(*num_points)++] = (Point3D){x1, y1, z1};
                stop_ray = voxel_map[x1 * submap_y * submap_z + y1 * submap_z + z1] != 0;}
                else stop_ray = true;
            }
        }
    }
    // Free any additional allocated memory here if necessary

    // Ensure the number of points does not exceed the allocated memory
    if (*num_points > max_points) {
        printf("Number of points exceeds allocated memory\n");
        free(points);
        return NULL;
    }
    return points;
}

// Free memory allocated for points
void FreePoints(Point3D *points) {
    free(points);
}
