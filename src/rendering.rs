use crate::{scene::{Camera, Scene, Color, Intersectable, Intersection, Light, Material, Coloration},
             geometry::Ray};
use image::{Rgb, RgbImage, Pixel};
use crate::vector::V3;
use rayon::prelude::*;

const SHADOW_BIAS : f64 = 1e-10;

pub fn render_depth(camera: &Camera, scene: &Scene) -> RgbImage {
    let ray_bundle = camera.get_ray_bundle();
    let camera_axis = camera.pose.r.R.get_col(2);

    let depth_buffer : Vec<f64> =
    (0..(camera.image_size.0 * camera.image_size.1))
    .into_par_iter()
    .map(|index| {
        let i = index / camera.image_size.1;
        let j = index % camera.image_size.1;
        let ray = ray_bundle[i][j];

        let mut z = f64::INFINITY;
        if let Some(intersection) = scene.cast(&ray) {
            z =  intersection.time_of_flight * V3::dot(ray.direction, camera_axis);
        }
        z
        }).collect();

    let max_depth = depth_buffer.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));
    let min_depth = depth_buffer.iter().fold(f64::INFINITY, |a, &b| a.min(b));
    let mut image = RgbImage::new(camera.image_size.0 as u32, camera.image_size.1 as u32);
    //display inverse_depth
    let min_inv_depth = 1.0 / max_depth;
    let max_inv_depth = 1.0 / min_depth;
    for i in 0..camera.image_size.0 {
        for j in 0..camera.image_size.1 {
            let depth = depth_buffer[i * camera.image_size.1 + j];
            let inv_depth = 1.0 / depth;
            let mapped_inv_depth = (inv_depth - min_inv_depth) / (max_inv_depth - min_inv_depth);
            let u8_inv_depth = (mapped_inv_depth * 255.0) as u8;
            let pixel = Rgb::from_channels(u8_inv_depth, u8_inv_depth, u8_inv_depth, 255);
            image.put_pixel(i as u32, j as u32, pixel);
        }
    }
    return image;
}

pub fn render_image(camera: &Camera, scene: &Scene, max_ray_recursion : usize) -> RgbImage {
    let ray_bundle = camera.get_ray_bundle();
    
    let pixel_buffer : Vec<Color> =
    (0..(camera.image_size.0 * camera.image_size.1))
    .into_par_iter()
    .map(|index| {
        let i = index / camera.image_size.1;
        let j = index % camera.image_size.1;
        let ray = ray_bundle[i][j];

        let pixel_color = get_ray_color(&ray, &scene, max_ray_recursion);
        pixel_color
    }).collect();

    let mut image = RgbImage::new(camera.image_size.0 as u32, camera.image_size.1 as u32);
    for i in 0..camera.image_size.0 {
        for j in 0..camera.image_size.1 {
            let color = pixel_buffer[i * camera.image_size.1 + j];
            image.put_pixel(i as u32, j as u32, color.into());
        }
    }
    return image;
}

fn get_ray_color(ray : &Ray, scene : &Scene, max_ray_recursion : usize) -> Color {
    let mut closest_tof = f64::INFINITY;
    let mut pixel_color  = Color::black();
    for element in &scene.elements {
        if let Some(intersection) = element.geometry.intersect(&ray) {                   
            if intersection.time_of_flight < closest_tof {
                let shadow_point = intersection.point + intersection.normal * SHADOW_BIAS;
                closest_tof = intersection.time_of_flight;
                
                match &element.material {
                    Material::Diffuse { albedo, coloration } => {
                        pixel_color = compute_diffuse_color(&intersection, albedo, coloration, &scene);
                    },
                    Material::Reflective { reflectivity, albedo, coloration } => {
                        let diffuse_color = compute_diffuse_color(&intersection, albedo, coloration, &scene);
                        if max_ray_recursion != 0 {
                            let reflection_ray = ray.reflect(shadow_point,intersection.normal);
                            let reflection_color = get_ray_color(&reflection_ray,
                                &scene,
                                 max_ray_recursion-1);
                            pixel_color = *reflectivity * reflection_color + (1.0 - *reflectivity) * diffuse_color;

                        } else {
                            pixel_color = (1.0 - *reflectivity) * diffuse_color;
                        }
                    },
                    Material::Glass {index, transparency} => {
                        let kr = fresnel(ray.direction, intersection.normal, *index);
                        if kr < 1.0 {
                            if let Some(transmission_ray) = ray.transmit(shadow_point, intersection.normal,
                                 *index, 1.0) {
                                    let refractive_color = get_ray_color(&transmission_ray,
                                                                     &scene,
                                                                    max_ray_recursion-1) * (1.0 - kr as f32);
                                 }
                        }
                    }
                }
            }
        }
    }
    pixel_color.clamp()
}

fn compute_diffuse_color(intersection : &Intersection, albedo : &f32, coloration : &Coloration,  scene : &Scene) -> Color {
    let mut pixel_color = Color::black();
    for light in &scene.lights {
        let direction_to_light : V3;
        let light_color : Color;
        let light_intensity : f32;
        match light {
            Light::Directional(dir_light) => {
                direction_to_light = -dir_light.direction;
                light_color = dir_light.color;
                light_intensity = dir_light.intensity;
            
                let shadow_ray = Ray::new(intersection.point + (intersection.normal * SHADOW_BIAS),
                direction_to_light);
                if let Some(_) = scene.cast(&shadow_ray) {continue;}
            
            }
            Light::Point(point_light) => {
                direction_to_light = (point_light.position - intersection.point).normalize();
                light_color = point_light.color;

                let r2 = (point_light.position - intersection.point).norm2() as f32;
                light_intensity = point_light.intensity / (4.0 * ::std::f32::consts::PI * r2);

                let shadow_ray = Ray::new(intersection.point + (intersection.normal * SHADOW_BIAS),
                direction_to_light);
                if let Some(shadow_intersection) = scene.cast(&shadow_ray) {
                    if shadow_intersection.time_of_flight < r2.sqrt() as f64 {continue;}
                    }
            }
        }

        pixel_color = pixel_color + lambret_cosine_law(intersection.normal,
                                direction_to_light,
                                light_intensity,
                                light_color,
                                coloration.color(intersection.texture_coords),
                                *albedo);
    }
    pixel_color

}

fn lambret_cosine_law(surface_normal : V3, direction_to_light :V3, light_intensity : f32, light_color : Color,
    element_color : Color, element_albedo : f32) -> Color {
    
    assert!(surface_normal.is_unit_length());
    assert!(direction_to_light.is_unit_length());
    
    let cos_theta = V3::dot(surface_normal, direction_to_light).max(0.0) as f32;
    let light_power = light_intensity * cos_theta;
    let light_reflected = element_albedo / std::f32::consts::PI;
    (element_color * light_color * light_power * light_reflected).clamp()
}

fn fresnel(incident: V3, normal: V3, index: f64) -> f64 {
    let i_dot_n = V3::dot(incident,normal);
    let mut eta_i = 1.0;
    let mut eta_t = index as f64;
    if i_dot_n > 0.0 {
        eta_i = eta_t;
        eta_t = 1.0;
    }

    let sin_t = eta_i / eta_t * (1.0 - i_dot_n * i_dot_n).max(0.0).sqrt();
    if sin_t > 1.0 {
        //Total internal reflection
        return 1.0;
    } else {
        let cos_t = (1.0 - sin_t * sin_t).max(0.0).sqrt();
        let cos_i = cos_t.abs();
        let r_s = ((eta_t * cos_i) - (eta_i * cos_t)) / ((eta_t * cos_i) + (eta_i * cos_t));
        let r_p = ((eta_i * cos_i) - (eta_t * cos_t)) / ((eta_i * cos_i) + (eta_t * cos_t));
        return (r_s * r_s + r_p * r_p) / 2.0;
    }
}