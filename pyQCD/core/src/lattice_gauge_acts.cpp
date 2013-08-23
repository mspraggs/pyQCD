#include <lattice.hpp>
#include <utils.hpp>

double Lattice::computeLocalWilsonAction(const int link[5])
{
  // Calculate the contribution to the Wilson action from the given link
  int planes[3];
  double Psum = 0.0;

  // Work out which dimension the link is in, since it'll be irrelevant here
  int j = 0;
  for (int i = 0; i < 4; ++i) {
    if (link[4] != i) {
      planes[j] = i;
      ++j;
    }    
  }

  // For each plane, calculate the two plaquettes that share the given link
  for (int i = 0; i < 3; ++i) {
    int site[4] = {link[0], link[1], link[2], link[3]};
    Psum += this->computePlaquette(site, link[4], planes[i]);
    site[planes[i]] -= 1;
    Psum += this->computePlaquette(site, link[4], planes[i]);
  }

  return -this->beta_ * Psum;
}



double Lattice::computeLocalRectangleAction(const int link[5])
{
  // Calculate contribution to improved action from given link

  // First contrbution is from standard Wilson action, so add that in
  double out = 5.0 / 3.0 * this->computeLocalWilsonAction(link);
  double Rsum = 0;

  int planes[3];

  // Work out which dimension the link is in, since it'll be irrelevant here
  int j = 0;
  for (int i = 0; i < 4; ++i) {
    if (link[4] != i) {
      planes[j] = i;
      ++j;
    }
  }
  
  for (int i = 0; i < 3; ++i) {
    int site[4] = {link[0], link[1], link[2], link[3]};
    // Add the six rectangles that contain the link
    Rsum += this->computeRectangle(site, link[4], planes[i]);
    site[link[4]] -= 1;
    Rsum += this->computeRectangle(site, link[4], planes[i]);

    site[link[4]] += 1;
    site[planes[i]] -= 1;
    Rsum += this->computeRectangle(site, link[4], planes[i]);
    site[link[4]] -= 1;
    Rsum += this->computeRectangle(site, link[4], planes[i]);
    
    site[link[4]] += 1;
    site[planes[i]] += 1;
    Rsum += this->computeRectangle(site, planes[i], link[4]);
    site[planes[i]] -= 2;
    Rsum += this->computeRectangle(site, planes[i], link[4]);
  }
  out += this->beta_ / (12 * pow(this->u0_, 2)) * Rsum;
  return out;
}



double Lattice::computeLocalTwistedRectangleAction(const int link[5])
{
  // Calculate contribution to improved action from given link

  // First contrbution is from standard Wilson action, so add that in
  double out = this->computeLocalWilsonAction(link);
  double Tsum = 0;

  int planes[3];

  // Work out which dimension the link is in, since it'll be irrelevant here
  int j = 0;
  for (int i = 0; i < 4; ++i) {
    if (link[4] != i) {
      planes[j] = i;
      ++j;
    }
  }
  
  for (int i = 0; i < 3; ++i) {
    int site[4] = {link[0], link[1], link[2], link[3]};
    // Add the seven twisted rectangles that contain the link
    Tsum += this->computeTwistedRectangle(site, link[4], planes[i]);
    site[link[4]] -= 1;
    Tsum += this->computeTwistedRectangle(site, link[4], planes[i]);

    site[link[4]] += 1;
    site[planes[i]] -= 1;
    Tsum += this->computeTwistedRectangle(site, link[4], planes[i]);
    site[link[4]] -= 1;
    Tsum += this->computeTwistedRectangle(site, link[4], planes[i]);
    
    site[link[4]] += 1;
    site[planes[i]] += 1;
    Tsum += this->computeTwistedRectangle(site, planes[i], link[4]);
    site[planes[i]] -= 1;
    Tsum += this->computeTwistedRectangle(site, planes[i], link[4]);
    site[planes[i]] -= 1;
    Tsum += this->computeTwistedRectangle(site, planes[i], link[4]);
  }
  out -= this->beta_ / (12 * pow(this->u0_, 4)) * Tsum;
  return out;
}



Matrix3cd Lattice::computeWilsonStaples(const int link[5])
{
  // Calculates the sum of staples for the two plaquettes surrounding
  // the link
  int planes[3];

  Matrix3cd out = Matrix3cd::Zero();
  
  // Work out which dimension the link is in, since it'll be irrelevant here
  int j = 0;
  for (int i = 0; i < 4; ++i) {
    if (link[4] != i) {
      planes[j] = i;
      ++j;
    }    
  }

  // For each plane, return the sum of the two link products for the
  // plaquette it resides in
  for (int i = 0; i < 3; ++i) {
    // Create a temporary link array to keep track of which link we're using
    int tempLink[5];
    // Initialise it
    copy(link, link + 5, tempLink);
    
    // First link is U_nu (x + mu)
    tempLink[4] = planes[i];
    tempLink[link[4]] += 1;
    Matrix3cd tempMatrix = this->getLink(tempLink);
    // Next link is U+_mu (x + nu)
    tempLink[4] = link[4];
    tempLink[link[4]] -= 1;
    tempLink[planes[i]] += 1;
    tempMatrix *= this->getLink(tempLink).adjoint();
    // Next link is U+_nu (x)
    tempLink[planes[i]] -= 1;
    tempLink[4] = planes[i];
    tempMatrix *= this->getLink(tempLink).adjoint();
    // And add it to the output
    out += tempMatrix;

    // First link is U+_nu (x + mu - nu)
    tempLink[link[4]] += 1;
    tempLink[planes[i]] -= 1;
    tempMatrix = this->getLink(tempLink).adjoint();
    // Next link is U+_mu (x - nu)
    tempLink[4] = link[4];
    tempLink[link[4]] -= 1;
    tempMatrix *= this->getLink(tempLink).adjoint();
    // Next link is U_nu (x - nu)
    tempLink[4] = planes[i];
    tempMatrix *= this->getLink(tempLink);
    // And add it to the output
    out += tempMatrix;
  }
  return out;
}



Matrix3cd Lattice::computeRectangleStaples(const int link[5])
{
  // Calculates the sum of staples for the six rectangles including
  // the link
  int planes[3];
  
  // Work out which dimension the link is in, since it'll be irrelevant here
  int j = 0;
  for (int i = 0; i < 4; ++i) {
    if (link[4] != i) {
      planes[j] = i;
      ++j;
    }    
  }

  Matrix3cd wilsonStaples = this->computeWilsonStaples(link);

  Matrix3cd rectangleStaples = Matrix3cd::Zero();

  // For each plane, return the sum of the two link products for the
  // plaquette it resides in
  for (int i = 0; i < 3; ++i) {
    // Create temporary array to keep track of links
    int tempLink[5];
    // Initialise it
    copy(link, link + 5, tempLink);
    // First link is U_mu (x + mu)
    tempLink[link[4]] += 1;
    Matrix3cd tempMatrix = this->getLink(tempLink);
    // Next link is U_nu (x + 2 * mu)
    tempLink[link[4]] += 1;
    tempLink[4] = planes[i];
    tempMatrix *= this->getLink(tempLink);
    // Next link U+_mu (x + mu + nu)
    tempLink[link[4]] -= 1;
    tempLink[planes[i]] += 1;
    tempLink[4] = link[4];
    tempMatrix *= this->getLink(tempLink).adjoint();
    // Next link is U+_mu (x + nu)
    tempLink[link[4]] -= 1;
    tempMatrix *= this->getLink(tempLink).adjoint();
    // Next link is U+_nu (x)
    tempLink[planes[i]] -= 1;
    tempLink[4] = planes[i];
    tempMatrix *= this->getLink(tempLink).adjoint();
    // Add it to the output
    rectangleStaples += tempMatrix;
    
    // Next is previous rectangle but translated by -1 in current plane
    // First link is U_mu (x + mu)
    tempLink[link[4]] += 1;
    tempLink[4] = link[4];
    tempMatrix = this->getLink(tempLink);
    // Next link is U+_nu (x + 2 * mu - nu)
    tempLink[link[4]] += 1;
    tempLink[planes[i]] -= 1;
    tempLink[4] = planes[i];
    tempMatrix *= this->getLink(tempLink).adjoint();
    // Next link U+_mu (x + mu - nu)
    tempLink[link[4]] -= 1;
    tempLink[4] = link[4];
    tempMatrix *= this->getLink(tempLink).adjoint();
    // Next link is U+_mu (x - nu)
    tempLink[link[4]] -= 1;
    tempMatrix *= this->getLink(tempLink).adjoint();
    // Next link is U_nu (x - nu)
    tempLink[4] = planes[i];
    tempMatrix *= this->getLink(tempLink);
    // Add it to the output
    rectangleStaples += tempMatrix;

    // Next is previous two rectangles but translated by -1 in link axis
    // First link is U_nu (x + mu)
    tempLink[link[4]] += 1;
    tempLink[planes[i]] += 1;
    tempMatrix = this->getLink(tempLink);
    // Next link is U+_mu (x + nu)
    tempLink[planes[i]] += 1;
    tempLink[link[4]] -= 1;
    tempLink[4] = link[4];
    tempMatrix *= this->getLink(tempLink).adjoint();
    // Next link is U+_mu (x - mu + nu)
    tempLink[link[4]] -= 1;
    tempMatrix *= this->getLink(tempLink).adjoint();
    // Next link is U+_nu (x - mu)
    tempLink[4] = planes[i];
    tempLink[planes[i]] -= 1;
    tempMatrix *= this->getLink(tempLink).adjoint();
    // Next link is U_mu (x - mu)
    tempLink[4] = link[4];
    tempMatrix *= this->getLink(tempLink);
    // Add it to the output
    rectangleStaples += tempMatrix;

    // Next is same rectangle but reflected in link axis
    // First link is U+_nu (x + mu - nu)
    tempLink[link[4]] += 2;
    tempLink[planes[i]] -= 1;
    tempLink[4] = planes[i];
    tempMatrix = this->getLink(tempLink).adjoint();
    // Next link is U+_mu (x - nu)
    tempLink[link[4]] -= 1;
    tempLink[4] = link[4];
    tempMatrix *= this->getLink(tempLink).adjoint();
    // Next link is U+_mu (x - mu - nu)
    tempLink[link[4]] -= 1;
    tempMatrix *= this->getLink(tempLink).adjoint();
    // Next link is U_nu (x - mu - nu)
    tempLink[4] = planes[i];
    tempMatrix *= this->getLink(tempLink);
    // Next link is U_mu (x - mu)
    tempLink[4] = link[4];
    tempLink[planes[i]] += 1;
    tempMatrix *= this->getLink(tempLink);
    // Add it to the output
    rectangleStaples += tempMatrix;

    // Next we do the rectangles rotated by 90 degrees
    // Link is U_nu (x + mu)
    tempLink[link[4]] += 2;
    tempLink[4] = planes[i];
    tempMatrix = this->getLink(tempLink);
    // Link is U_nu (x + mu + nu)
    tempLink[planes[i]] += 1;
    tempMatrix *= this->getLink(tempLink);
    // Link is U+_mu (x + 2 * nu)
    tempLink[4] = link[4];
    tempLink[link[4]] -= 1;
    tempLink[planes[i]] += 1;
    tempMatrix *= this->getLink(tempLink).adjoint();
    // Link is U+_nu (x + nu)
    tempLink[4] = planes[i];
    tempLink[planes[i]] -= 1;
    tempMatrix *= this->getLink(tempLink).adjoint();
    // Link is U+_nu (x)
    tempLink[planes[i]] -= 1;
    tempMatrix *= this->getLink(tempLink).adjoint();
    // Add to the sum
    rectangleStaples += tempMatrix;

    // Next flip the previous rectangle across the link axis
    // Link is U+_nu (x + mu - nu)
    tempLink[link[4]] += 1;
    tempLink[planes[i]] -= 1;
    tempLink[4] = planes[i];
    tempMatrix = this->getLink(tempLink).adjoint();
    // Link is U+_nu (x + mu - 2 * nu)
    tempLink[planes[i]] -= 1;
    tempMatrix *= this->getLink(tempLink).adjoint();
    // Link is U+_mu (x - 2 * nu)
    tempLink[4] = link[4];
    tempLink[link[4]] -= 1;
    tempMatrix *= this->getLink(tempLink).adjoint();
    // Link is U_nu (x - 2 * nu)
    tempLink[4] = planes[i];
    tempMatrix *= this->getLink(tempLink);
    // Link is U_nu (x - nu)
    tempLink[planes[i]] += 1;
    tempMatrix *= this->getLink(tempLink);
    // Add to the sum
    rectangleStaples += tempMatrix;
  }

  return 5.0 / 3.0 * wilsonStaples 
    - rectangleStaples / (12.0 * pow(this->u0_, 2));
}



Matrix3cd Lattice::computeTwistedRectangleStaples(const int link[5])
{
  cout << "Error! Cannot compute sum of staples for twisted rectangle "
       << "operator." << endl;
  return Matrix3cd::Identity();
}
