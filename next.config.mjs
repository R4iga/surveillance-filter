/** @type {import('next').NextConfig} */
const nextConfig = {
  reactStrictMode: true,
  transpilePackages: ['@tensorflow/tfjs', '@tensorflow/tfjs-backend-cpu', '@tensorflow/tfjs-backend-webgl', '@tensorflow/tfjs-converter', '@tensorflow/tfjs-core', '@tensorflow-models/coco-ssd'],
};
export default nextConfig;
